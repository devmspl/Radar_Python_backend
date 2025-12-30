from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import os
import json
import logging
from typing import List, Optional, Dict, Any
from database import get_db
from models import Quiz, QuizCategory, UserQuizScore, Feed, Blog, Transcript, User
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from schemas import QuizCategoryResponse,QuizResponse,QuizResultResponse,QuizSubmission,UserQuizHistory
from sqlalchemy import func

router = APIRouter(prefix="/quizzes", tags=["Quizzes"])

logger = logging.getLogger(__name__)
openai_api_key = os.getenv("OPENAI_API_KEY")
client = None
if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
else:
    logger.warning("OPENAI_API_KEY not found. AI quiz generation will be disabled.")

# Initialize scheduler
scheduler = BackgroundScheduler()

# ------------------ Quiz Categories ------------------

DEFAULT_CATEGORIES = [
    {"name": "UI/UX Design", "description": "User interface and experience design principles"},
    {"name": "Product Management", "description": "Product development and management strategies"},
    {"name": "Science", "description": "Scientific concepts and discoveries"},
    {"name": "Marketing", "description": "Marketing strategies and digital marketing"},
    {"name": "Writing", "description": "Content writing and communication skills"},
    {"name": "Lifestyle", "description": "Health, wellness, and lifestyle topics"},
    {"name": "Consulting & HR", "description": "Consulting strategies and human resources"},
    {"name": "Photography", "description": "Photography techniques and composition"},
    {"name": "Coding", "description": "Programming and software development"}
]

def initialize_categories(db: Session):
    """Initialize default quiz categories"""
    for cat_data in DEFAULT_CATEGORIES:
        existing = db.query(QuizCategory).filter(QuizCategory.name == cat_data["name"]).first()
        if not existing:
            category = QuizCategory(
                name=cat_data["name"],
                description=cat_data["description"],
                is_active=True
            )
            db.add(category)
    db.commit()

# ------------------ AI Quiz Generation ------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_quiz_with_ai(category: str, content: str, difficulty: str) -> Dict[str, Any]:
    """Generate 10 quiz questions using OpenAI."""
    try:
        truncated_content = content[:12000] + "..." if len(content) > 12000 else content
        
        prompt = f"""
        Create a {difficulty} difficulty quiz about {category} with EXACTLY 10 multiple-choice questions.
        
        Requirements:
        - Generate exactly 10 questions (no more, no less)
        - Each question must have 4 options (A, B, C, D)
        - Mark the correct answer clearly (0-3 index)
        - Include brief explanations for each answer
        - Questions should test understanding of key concepts from the content
        - Make questions relevant to the specific content provided
        - Vary question types: conceptual, practical, scenario-based
        
        Content:
        {truncated_content}
        
        Return JSON format:
        {{
            "title": "Engaging Quiz Title about {category}",
            "description": "Brief description of what this quiz covers",
            "questions": [
                {{
                    "question": "Clear and concise question text",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": 0,
                    "explanation": "Brief explanation why this is correct"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a quiz creation expert. Create engaging, educational multiple-choice quizzes. Always return valid JSON with exactly 10 questions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        content = response.choices[0].message.content
        quiz_data = json.loads(content)
        
        # Validate we have exactly 10 questions
        questions = quiz_data.get("questions", [])
        if len(questions) != 10:
            raise ValueError(f"Expected 10 questions, got {len(questions)}")
        
        return quiz_data
        
    except Exception as e:
        logger.error(f"OpenAI quiz generation error: {e}")
        return generate_fallback_quiz(category, difficulty)

def generate_fallback_quiz(category: str, difficulty: str) -> Dict[str, Any]:
    """Generate fallback quiz with 10 questions."""
    questions = []
    for i in range(10):
        questions.append({
            "question": f"Question {i+1}: What is an important aspect of {category}?",
            "options": [
                f"Fundamental concept A in {category}",
                f"Key principle B in {category}",
                f"Core technique C in {category}",
                f"Essential strategy D in {category}"
            ],
            "correct_answer": i % 4,  # Rotate correct answers
            "explanation": f"This covers important aspects of {category} that professionals should know."
        })
    
    return {
        "title": f"{category} Knowledge Test",
        "description": f"Assess your understanding of {category} concepts",
        "questions": questions
    }

def get_content_for_category(db: Session, category_id: int) -> str:
    """Get relevant content for quiz generation from blogs and transcripts."""
    category = db.query(QuizCategory).filter(QuizCategory.id == category_id).first()
    if not category:
        return ""
    
    content_parts = []
    category_name = category.name

    # 🧠 FIXED: use .like instead of .contains([category_name])
    blogs = db.query(Blog).filter(
        Blog.category.like(f"%{category_name}%")
    ).limit(5).all()
    
    for blog in blogs:
        content_parts.append(f"BLOG: {blog.title}\n{blog.content[:2500]}...")
    
    transcripts = db.query(Transcript).all()
    
    for transcript in transcripts:
        content_parts.append(f"VIDEO: {transcript.title}\n{transcript.transcript_text[:2500]}...")
    
    feeds = db.query(Feed).filter(
        Feed.categories.like(f"%{category_name}%")
    ).limit(5).all()
    
    for feed in feeds:
        if feed.source_type == "blog" and feed.blog:
            content_parts.append(f"BLOG FEED: {feed.title}\n{feed.blog.content[:2000]}...")
        elif feed.source_type == "youtube":
            ai_content = getattr(feed, 'ai_generated_content', {})
            summary = ai_content.get('summary', '')
            key_points = ai_content.get('key_points', [])
            content_parts.append(f"VIDEO FEED: {feed.title}\nSummary: {summary}\nKey Points: {'; '.join(key_points[:5])}")
    
    return "\n\n".join(content_parts) if content_parts else f"General knowledge and best practices in {category_name}"

def create_quiz_for_category(db: Session, category_id: int, difficulty: str = "medium") -> Quiz:
    """Create a new quiz for a category."""
    try:
        category = db.query(QuizCategory).filter(QuizCategory.id == category_id).first()
        if not category:
            raise ValueError(f"Category {category_id} not found")
        
        # Get content for quiz generation
        content = get_content_for_category(db, category_id)
        
        if not content or len(content.strip()) < 500:
            logger.warning(f"Insufficient content for {category.name} quiz, using fallback")
            content = f"Comprehensive overview of {category.name} including key concepts, best practices, and industry standards."
        
        # Generate quiz using AI
        quiz_data = generate_quiz_with_ai(category.name, content, difficulty)
        
        # Deactivate old quizzes for this category
        db.query(Quiz).filter(
            Quiz.category_id == category_id,
            Quiz.is_active == True
        ).update({"is_active": False})
        
        # Create new quiz
        quiz = Quiz(
            title=quiz_data["title"],
            description=quiz_data.get("description", f"Test your knowledge of {category.name}"),
            category_id=category_id,
            difficulty=difficulty,
            questions=quiz_data["questions"],
            source_type="mixed",
            is_active=True,
            version=1,
            last_updated=datetime.utcnow()
        )
        
        db.add(quiz)
        db.commit()
        db.refresh(quiz)
        
        logger.info(f"✅ Created quiz {quiz.id} for category {category.name}")
        return quiz
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error creating quiz for category {category_id}: {e}")
        raise
def generate_quiz_variations(db: Session, category_id: int, num_quizzes: int = 3):
    """Generate multiple quiz variations for a category"""
    category = db.query(QuizCategory).filter(QuizCategory.id == category_id).first()
    if not category:
        raise ValueError(f"Category {category_id} not found")
    
    # Get content for quiz generation
    content = get_content_for_category(db, category_id)
    
    if not content or len(content.strip()) < 500:
        logger.warning(f"Insufficient content for {category.name} quizzes, using fallback")
        content = f"Comprehensive overview of {category.name} including key concepts, best practices, and industry standards."
    
    quizzes_created = []
    
    for i in range(num_quizzes):
        try:
            # Use different difficulties or focuses for variety
            difficulties = ["easy", "medium", "hard"]
            difficulty = difficulties[i % len(difficulties)]
            
            # Add variation to the prompt
            variations = [
                "focus on fundamental concepts and basics",
                "focus on practical applications and real-world scenarios", 
                "focus on advanced techniques and industry best practices",
                "focus on common challenges and problem-solving",
                "focus on emerging trends and future developments"
            ]
            
            variation_prompt = variations[i % len(variations)]
            
            # Generate unique quiz
            quiz_data = generate_quiz_with_variation(category.name, content, difficulty, variation_prompt)
            
            # Deactivate only if we're replacing existing ones
            if i == 0:  # Only deactivate for the first quiz if replacing
                db.query(Quiz).filter(
                    Quiz.category_id == category_id,
                    Quiz.is_active == True
                ).update({"is_active": False})
            
            # Create new quiz
            quiz = Quiz(
                title=f"{quiz_data['title']} - {variation_prompt.split(' ')[1].title()}",
                description=quiz_data.get("description", f"Test your {variation_prompt} knowledge of {category.name}"),
                category_id=category_id,
                difficulty=difficulty,
                questions=quiz_data["questions"],
                source_type="mixed",
                is_active=True,
                version=i + 1,
                last_updated=datetime.utcnow()
            )
            
            db.add(quiz)
            quizzes_created.append(quiz)
            
            logger.info(f"✅ Created quiz variation {i+1} for category {category.name}")
            
        except Exception as e:
            logger.error(f"❌ Error creating quiz variation {i+1} for category {category_id}: {e}")
            continue
    
    db.commit()
    
    # Refresh all created quizzes
    for quiz in quizzes_created:
        db.refresh(quiz)
    
    return quizzes_created

def generate_quiz_with_variation(category: str, content: str, difficulty: str, variation: str) -> Dict[str, Any]:
    """Generate quiz with specific variation"""
    try:
        truncated_content = content[:12000] + "..." if len(content) > 12000 else content
        
        prompt = f"""
        Create a {difficulty} difficulty quiz about {category} with EXACTLY 10 multiple-choice questions.
        
        Specific Focus: {variation}
        
        Requirements:
        - Generate exactly 10 questions (no more, no less)
        - Each question must have 4 options (A, B, C, D)
        - Mark the correct answer clearly (0-3 index)
        - Include brief explanations for each answer
        - Questions should test understanding of key concepts from the content
        - Make questions relevant to the specific focus and content provided
        - Vary question types: conceptual, practical, scenario-based
        
        Content:
        {truncated_content}
        
        Return JSON format:
        {{
            "title": "Engaging Quiz Title about {category} - {variation}",
            "description": "Brief description focusing on {variation}",
            "questions": [
                {{
                    "question": "Clear and concise question text",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": 0,
                    "explanation": "Brief explanation why this is correct"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a quiz creation expert. Create engaging, educational multiple-choice quizzes. Always return valid JSON with exactly 10 questions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8,  # Higher temperature for more variation
            max_tokens=4000
        )
        
        content = response.choices[0].message.content
        quiz_data = json.loads(content)
        
        # Validate we have exactly 10 questions
        questions = quiz_data.get("questions", [])
        if len(questions) != 10:
            raise ValueError(f"Expected 10 questions, got {len(questions)}")
        
        return quiz_data
        
    except Exception as e:
        logger.error(f"OpenAI quiz generation error: {e}")
        return generate_fallback_quiz(category, difficulty)
# ------------------ Quiz Update Scheduler ------------------

def update_quizzes_job():
    """Background job to update quizzes with multiple variations"""
    from database import SessionLocal
    db = SessionLocal()
    try:
        logger.info("🔄 Starting daily quiz update with multiple variations...")
        
        # Get all active categories
        categories = db.query(QuizCategory).filter(QuizCategory.is_active == True).all()
        
        updated_count = 0
        for category in categories:
            try:
                # Check if category needs update (quizzes older than 24 hours or less than 3 quizzes)
                active_quizzes = db.query(Quiz).filter(
                    Quiz.category_id == category.id,
                    Quiz.is_active == True
                ).all()
                
                needs_update = False
                
                if not active_quizzes:
                    needs_update = True
                elif len(active_quizzes) < 3:
                    needs_update = True
                else:
                    # Check if any quiz is older than 24 hours
                    latest_quiz = max(active_quizzes, key=lambda q: q.last_updated)
                    if (datetime.utcnow() - latest_quiz.last_updated).total_seconds() > 86400:
                        needs_update = True
                
                if needs_update:
                    # Generate 3 quiz variations
                    generate_quiz_variations(db, category.id, 3)
                    updated_count += 1
                    logger.info(f"🔄 Updated quizzes for {category.name}")
                else:
                    logger.info(f"⏭️  Quizzes for {category.name} are up to date")
                    
            except Exception as e:
                logger.error(f"❌ Failed to update quizzes for {category.name}: {e}")
                continue
        
        logger.info(f"✅ Daily quiz update completed: {updated_count} categories updated")
        
    except Exception as e:
        logger.error(f"❌ Quiz update job failed: {e}")
    finally:
        db.close()

def start_scheduler():
    """Start the background scheduler for quiz updates"""
    if not scheduler.running:
        # Update quizzes every day at 2 AM
        scheduler.add_job(
            update_quizzes_job,
            trigger=CronTrigger(hour=2, minute=0),
            id='daily_quiz_update',
            replace_existing=True
        )
        scheduler.start()
        logger.info("✅ Quiz update scheduler started")

# ------------------ API Endpoints ------------------

@router.on_event("startup")
async def startup_event():
    """Initialize categories and start scheduler on startup"""
    db = next(get_db())
    try:
        initialize_categories(db)
        start_scheduler()
    finally:
        db.close()

@router.get("/categories", response_model=List[QuizCategoryResponse])
def get_categories(db: Session = Depends(get_db)):
    """Get all quiz categories with quiz counts"""
    categories = db.query(QuizCategory).filter(QuizCategory.is_active == True).all()
    
    result = []
    for category in categories:
        quiz_count = db.query(Quiz).filter(
            Quiz.category_id == category.id,
            Quiz.is_active == True
        ).count()
        
        result.append({
            "id": category.id,
            "name": category.name,
            "description": category.description,
            "is_active": category.is_active,
            "quiz_count": quiz_count,
            "created_at": category.created_at.isoformat() if category.created_at else None
        })
    
    return result
@router.post("/category/{category_id}/generate-multiple")
def generate_multiple_quizzes(
    category_id: int,
    num_quizzes: int = 3,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Generate multiple quizzes for a category"""
    category = db.query(QuizCategory).filter(QuizCategory.id == category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    if background_tasks:
        # Run in background for large numbers
        def generate_task():
            db_local = next(get_db())
            try:
                generate_quiz_variations(db_local, category_id, num_quizzes)
            finally:
                db_local.close()
        
        background_tasks.add_task(generate_task)
        return {"message": f"Generating {num_quizzes} quizzes for {category.name} in background"}
    else:
        # Generate immediately
        quizzes = generate_quiz_variations(db, category_id, num_quizzes)
        return {
            "message": f"Generated {len(quizzes)} quizzes for {category.name}",
            "quiz_ids": [quiz.id for quiz in quizzes]
        }

@router.get("/category/{category_id}/quizzes")
def get_all_quizzes_by_category(
    category_id: int,
    include_questions: bool = False,
    db: Session = Depends(get_db)
):
    """Get all quizzes for a category"""
    category = db.query(QuizCategory).filter(QuizCategory.id == category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    quizzes = db.query(Quiz).filter(
        Quiz.category_id == category_id,
        Quiz.is_active == True
    ).order_by(Quiz.last_updated.desc()).all()
    
    quizzes_list = []
    for quiz in quizzes:
        quiz_data = {
            "id": quiz.id,
            "title": quiz.title,
            "description": quiz.description,
            "difficulty": quiz.difficulty,
            "version": quiz.version,
            "question_count": len(quiz.questions) if quiz.questions else 0,
            "last_updated": quiz.last_updated.isoformat() if quiz.last_updated else None
        }
        
        if include_questions:
            quiz_data["questions"] = quiz.questions
        
        quizzes_list.append(quiz_data)
    
    return {
        "category_id": category_id,
        "category_name": category.name,
        "total_quizzes": len(quizzes_list),
        "quizzes": quizzes_list
    }

@router.get("/{quiz_id}", response_model=Dict[str, Any])
def get_quiz_by_id(
    quiz_id: int,
    user_id: int = 1,  # In real app, get from auth token
    db: Session = Depends(get_db)
):
    """Get complete quiz details with all questions in the specified format"""
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    # Get category details
    category = db.query(QuizCategory).filter(QuizCategory.id == quiz.category_id).first()
    
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Transform questions to the desired format
    formatted_questions = []
    for i, question in enumerate(quiz.questions):
        formatted_options = []
        for j, option_text in enumerate(question["options"]):
            formatted_options.append({
                "id": chr(97 + j),  # 'a', 'b', 'c', 'd'
                "text": option_text,
                "isCorrect": j == question["correct_answer"]
            })
        
        formatted_questions.append({
            "id": f"q{i+1}",
            "question": question["question"],
            "options": formatted_options,
            "explanation": question.get("explanation", "No explanation provided.")
        })
    
    # Create the response in the exact format you want
    response = {
        f"{category.id}-{quiz.id}": {
            "id": f"{category.id}-{quiz.id}",
            "title": quiz.title,
            "questions": formatted_questions
        }
    }
    
    return response

@router.post("/submit", response_model=QuizResultResponse)
def submit_quiz(
    submission: QuizSubmission,
    user_id: int = 1,  # In real app, get from auth token
    db: Session = Depends(get_db)
):
    """Submit quiz answers and calculate score"""
    quiz = db.query(Quiz).filter(Quiz.id == submission.quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    questions = quiz.questions
    user_answers = submission.answers
    
    # Calculate score
    correct_count = 0
    results = []
    
    for i, question in enumerate(questions):
        user_answer = user_answers.get(i)
        is_correct = user_answer == question["correct_answer"]
        
        if is_correct:
            correct_count += 1
        
        results.append({
            "question_index": i,
            "question": question["question"],
            "user_answer": user_answer,
            "correct_answer": question["correct_answer"],
            "is_correct": is_correct,
            "explanation": question.get("explanation", ""),
            "options": question["options"]
        })
    
    score = (correct_count / len(questions)) * 100
    
    # Save user score
    user_score = UserQuizScore(
        user_id=user_id,
        quiz_id=quiz.id,
        score=score,
        correct_answers=correct_count,
        total_questions=len(questions),
        time_taken=submission.time_taken,
        answers=user_answers,
        completed_at=datetime.utcnow()
    )
    
    db.add(user_score)
    db.commit()
    
    # Calculate user rank
    all_scores = db.query(UserQuizScore).filter(
        UserQuizScore.quiz_id == quiz.id
    ).all()
    
    user_rank = "average"
    if all_scores:
        scores = [s.score for s in all_scores]
        user_percentile = sum(1 for s in scores if s < score) / len(scores) * 100
        if user_percentile >= 90:
            user_rank = "top 10%"
        elif user_percentile >= 75:
            user_rank = "top 25%"
        elif user_percentile >= 50:
            user_rank = "above average"
    
    return {
        "quiz_id": quiz.id,
        "quiz_title": quiz.title,
        "score": round(score, 2),
        "correct_answers": correct_count,
        "total_questions": len(questions),
        "time_taken": submission.time_taken,
        "passed": score >= 70,
        "results": results,
        "user_rank": user_rank
    }

@router.get("/user/scores", response_model=List[UserQuizHistory])
def get_user_scores(
    user_id: int = 1,  # In real app, get from auth token
    category_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get user's quiz history"""
    query = db.query(UserQuizScore).join(Quiz).filter(UserQuizScore.user_id == user_id)
    
    if category_id:
        query = query.filter(Quiz.category_id == category_id)
    
    scores = query.order_by(UserQuizScore.completed_at.desc()).limit(50).all()
    
    result = []
    for score in scores:
        result.append({
            "quiz_id": score.quiz_id,
            "quiz_title": score.quiz.title,
            "category": score.quiz.category.name,
            "score": score.score,
            "completed_at": score.completed_at.isoformat(),
            "time_taken": score.time_taken
        })
    
    return result

@router.post("/category/{category_id}/refresh")
def refresh_category_quiz(
    category_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Manually refresh quiz for a category"""
    category = db.query(QuizCategory).filter(QuizCategory.id == category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    def refresh_task():
        db_local = next(get_db())
        try:
            create_quiz_for_category(db_local, category_id)
        finally:
            db_local.close()
    
    background_tasks.add_task(refresh_task)
    
    return {"message": f"Quiz refresh started for {category.name}"}
    
def get_username(db: Session, user_id: int) -> str:
    """Helper function to get user's display name"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        # Combine full_name and last_name, or use just full_name if last_name is empty
        if user.last_name:
            return f"{user.full_name} {user.last_name}".strip()
        else:
            return user.full_name
    else:
        return f"User_{user_id}"

@router.get("/leaderboard/overall")
def get_overall_leaderboard(
    timeframe: str = "all_time",  # "today", "weekly", "monthly", "all_time"
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get overall leaderboard across all categories"""
    
    # Base query
    query = db.query(UserQuizScore)
    
    # Apply timeframe filter
    if timeframe == "today":
        today = datetime.utcnow().date()
        query = query.filter(func.date(UserQuizScore.completed_at) == today)
    elif timeframe == "weekly":
        week_ago = datetime.utcnow() - timedelta(days=7)
        query = query.filter(UserQuizScore.completed_at >= week_ago)
    elif timeframe == "monthly":
        month_ago = datetime.utcnow() - timedelta(days=30)
        query = query.filter(UserQuizScore.completed_at >= month_ago)
    
    # Get top scores
    top_scores = query.order_by(
        UserQuizScore.score.desc()
    ).limit(limit).all()
    
    leaderboard = []
    for i, score in enumerate(top_scores, 1):
        # Get user and category info
        user = db.query(User).filter(User.id == score.user_id).first()
        category = db.query(QuizCategory).join(Quiz).filter(
            Quiz.id == score.quiz_id
        ).first()
        
        leaderboard.append({
            "rank": i,
            "user_id": score.user_id,
            "username": get_username(db, score.user_id),
            "email": user.email if user else None,
            "score": round(score.score, 2),
            "category": category.name if category else "Unknown",
            "quiz_title": score.quiz.title,
            "correct_answers": score.correct_answers,
            "total_questions": score.total_questions,
            "time_taken": score.time_taken,
            "completed_at": score.completed_at.isoformat(),
            "medal": get_medal(i)
        })
    
    return {
        "timeframe": timeframe,
        "limit": limit,
        "total_records": len(leaderboard),
        "leaderboard": leaderboard
    }

@router.get("/leaderboard/today-top")
def get_today_top_scorers(
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """Get top scorers for today"""
    
    today = datetime.utcnow().date()
    
    # Get today's top scores
    today_scores = db.query(UserQuizScore).filter(
        func.date(UserQuizScore.completed_at) == today
    ).order_by(
        UserQuizScore.score.desc()
    ).limit(limit).all()
    
    top_scorers = []
    for i, score in enumerate(today_scores, 1):
        user = db.query(User).filter(User.id == score.user_id).first()
        category = db.query(QuizCategory).join(Quiz).filter(
            Quiz.id == score.quiz_id
        ).first()
        
        top_scorers.append({
            "rank": i,
            "user_id": score.user_id,
            "username": get_username(db, score.user_id),
            "email": user.email if user else None,
            "score": round(score.score, 2),
            "category": category.name if category else "Unknown",
            "quiz_title": score.quiz.title,
            "correct_answers": score.correct_answers,
            "total_questions": score.total_questions,
            "completed_at": score.completed_at.isoformat(),
            "medal": get_medal(i)
        })
    
    # Get today's statistics
    today_stats = db.query(
        func.count(UserQuizScore.id).label('total_attempts'),
        func.count(func.distinct(UserQuizScore.user_id)).label('unique_users'),
        func.avg(UserQuizScore.score).label('average_score')
    ).filter(
        func.date(UserQuizScore.completed_at) == today
    ).first()
    
    return {
        "date": today.isoformat(),
        "total_attempts": today_stats.total_attempts or 0,
        "unique_users": today_stats.unique_users or 0,
        "average_score": round(today_stats.average_score, 2) if today_stats.average_score else 0,
        "top_scorers": top_scorers
    }

@router.get("/leaderboard/{category_id}")
def get_category_leaderboard(
    category_id: int,
    db: Session = Depends(get_db)
):
    """Get leaderboard for a specific category"""
    
    # Get top scores for this category
    top_scores = db.query(UserQuizScore).join(Quiz).filter(
        Quiz.category_id == category_id
    ).order_by(UserQuizScore.score.desc()).limit(10).all()
    
    category = db.query(QuizCategory).filter(QuizCategory.id == category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    leaderboard = []
    for i, score in enumerate(top_scores, 1):
        user = db.query(User).filter(User.id == score.user_id).first()
        
        leaderboard.append({
            "rank": i,
            "user_id": score.user_id,
            "username": get_username(db, score.user_id),
            "email": user.email if user else None,
            "score": round(score.score, 2),
            "correct_answers": score.correct_answers,
            "total_questions": score.total_questions,
            "time_taken": score.time_taken,
            "completed_at": score.completed_at.isoformat(),
            "medal": get_medal(i)
        })
    
    return {
        "category_id": category_id,
        "category_name": category.name,
        "total_players": len(leaderboard),
        "leaderboard": leaderboard
    }

def get_medal(rank: int) -> str:
    """Get medal emoji based on rank"""
    if rank == 1:
        return "🥇"
    elif rank == 2:
        return "🥈"
    elif rank == 3:
        return "🥉"
    else:
        return "🎯"

@router.get("/user/{user_id}/stats")
def get_user_stats(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed statistics for a specific user"""
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Overall stats
    overall_stats = db.query(
        func.count(UserQuizScore.id).label('total_quizzes'),
        func.avg(UserQuizScore.score).label('average_score'),
        func.max(UserQuizScore.score).label('best_score'),
        func.min(UserQuizScore.score).label('worst_score'),
        func.sum(UserQuizScore.correct_answers).label('total_correct'),
        func.sum(UserQuizScore.total_questions).label('total_questions')
    ).filter(
        UserQuizScore.user_id == user_id
    ).first()
    
    # Category-wise performance
    category_stats = db.query(
        QuizCategory.name,
        func.count(UserQuizScore.id).label('quiz_count'),
        func.avg(UserQuizScore.score).label('average_score'),
        func.max(UserQuizScore.score).label('best_score')
    ).join(
        Quiz, Quiz.id == UserQuizScore.quiz_id
    ).join(
        QuizCategory, QuizCategory.id == Quiz.category_id
    ).filter(
        UserQuizScore.user_id == user_id
    ).group_by(
        QuizCategory.name
    ).all()
    
    # Recent activity
    recent_scores = db.query(
        UserQuizScore.score,
        Quiz.title,
        QuizCategory.name,
        UserQuizScore.completed_at
    ).join(
        Quiz, Quiz.id == UserQuizScore.quiz_id
    ).join(
        QuizCategory, QuizCategory.id == Quiz.category_id
    ).filter(
        UserQuizScore.user_id == user_id
    ).order_by(
        UserQuizScore.completed_at.desc()
    ).limit(5).all()
    
    # Calculate accuracy
    total_questions = overall_stats.total_questions or 1
    total_correct = overall_stats.total_correct or 0
    accuracy = (total_correct / total_questions) * 100
    
    return {
        "user_id": user_id,
        "user_info": {
            "full_name": user.full_name,
            "last_name": user.last_name,
            "email": user.email
        },
        "overall_stats": {
            "total_quizzes": overall_stats.total_quizzes or 0,
            "average_score": round(overall_stats.average_score, 2) if overall_stats.average_score else 0,
            "best_score": round(overall_stats.best_score, 2) if overall_stats.best_score else 0,
            "worst_score": round(overall_stats.worst_score, 2) if overall_stats.worst_score else 0,
            "accuracy": round(accuracy, 2),
            "total_correct_answers": total_correct,
            "total_questions_attempted": total_questions
        },
        "category_performance": [
            {
                "category": stat.name,
                "quiz_count": stat.quiz_count,
                "average_score": round(stat.average_score, 2) if stat.average_score else 0,
                "best_score": round(stat.best_score, 2) if stat.best_score else 0
            }
            for stat in category_stats
        ],
        "recent_activity": [
            {
                "quiz_title": score.title,
                "category": score.name,
                "score": round(score.score, 2),
                "completed_at": score.completed_at.isoformat()
            }
            for score in recent_scores
        ]
    }

def get_performance_tier(score: float) -> str:
    """Determine performance tier based on average score"""
    if score >= 90:
        return "expert"
    elif score >= 80:
        return "advanced"
    elif score >= 70:
        return "intermediate"
    elif score >= 60:
        return "beginner"
    else:
        return "novice"

@router.get("/stats/overall")
def get_overall_quiz_stats(db: Session = Depends(get_db)):
    """Get comprehensive overall quiz statistics"""
    
    # Total quizzes taken
    total_quizzes_taken = db.query(func.count(UserQuizScore.id)).scalar() or 0
    
    # Total unique users who took quizzes
    total_unique_users = db.query(func.count(func.distinct(UserQuizScore.user_id))).scalar() or 0
    
    # Average score across all quizzes
    average_score = db.query(func.avg(UserQuizScore.score)).scalar() or 0
    
    # Total questions answered
    total_questions_answered = db.query(func.sum(UserQuizScore.total_questions)).scalar() or 0
    
    # Total correct answers
    total_correct_answers = db.query(func.sum(UserQuizScore.correct_answers)).scalar() or 0
    
    # Overall accuracy
    overall_accuracy = (total_correct_answers / total_questions_answered * 100) if total_questions_answered > 0 else 0
    
    # Category-wise statistics
    category_stats = db.query(
        QuizCategory.name,
        QuizCategory.id,
        func.count(UserQuizScore.id).label('quiz_count'),
        func.avg(UserQuizScore.score).label('avg_score'),
        func.count(func.distinct(UserQuizScore.user_id)).label('unique_users')
    ).join(
        Quiz, Quiz.id == UserQuizScore.quiz_id
    ).join(
        QuizCategory, QuizCategory.id == Quiz.category_id
    ).group_by(
        QuizCategory.name, QuizCategory.id
    ).all()
    
    # Daily activity (last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    daily_activity = db.query(
        func.date(UserQuizScore.completed_at).label('date'),
        func.count(UserQuizScore.id).label('quiz_count'),
        func.avg(UserQuizScore.score).label('avg_score'),
        func.count(func.distinct(UserQuizScore.user_id)).label('unique_users')
    ).filter(
        UserQuizScore.completed_at >= seven_days_ago
    ).group_by(
        func.date(UserQuizScore.completed_at)
    ).order_by(
        func.date(UserQuizScore.completed_at).desc()
    ).all()
    
    # Top performers (users with highest average scores, min 5 quizzes)
    top_performers = db.query(
        UserQuizScore.user_id,
        User.full_name,
        User.last_name,
        func.avg(UserQuizScore.score).label('avg_score'),
        func.count(UserQuizScore.id).label('quiz_count'),
        func.max(UserQuizScore.score).label('best_score')
    ).join(
        User, User.id == UserQuizScore.user_id
    ).group_by(
        UserQuizScore.user_id, User.full_name, User.last_name
    ).having(
        func.count(UserQuizScore.id) >= 3  # At least 3 quizzes
    ).order_by(
        func.avg(UserQuizScore.score).desc()
    ).limit(10).all()
    
    # Most popular categories (by number of attempts)
    popular_categories = db.query(
        QuizCategory.name,
        QuizCategory.id,
        func.count(UserQuizScore.id).label('attempt_count'),
        func.avg(UserQuizScore.score).label('avg_score')
    ).join(
        Quiz, Quiz.id == UserQuizScore.quiz_id
    ).join(
        QuizCategory, QuizCategory.id == Quiz.category_id
    ).group_by(
        QuizCategory.name, QuizCategory.id
    ).order_by(
        func.count(UserQuizScore.id).desc()
    ).limit(5).all()
    
    # Difficulty level statistics
    difficulty_stats = db.query(
        Quiz.difficulty,
        func.count(UserQuizScore.id).label('quiz_count'),
        func.avg(UserQuizScore.score).label('avg_score'),
        func.avg(UserQuizScore.time_taken).label('avg_time_taken')
    ).join(
        Quiz, Quiz.id == UserQuizScore.quiz_id
    ).group_by(
        Quiz.difficulty
    ).all()
    
    # Time-based statistics
    today = datetime.utcnow().date()
    today_stats = db.query(
        func.count(UserQuizScore.id).label('today_quizzes'),
        func.avg(UserQuizScore.score).label('today_avg_score'),
        func.count(func.distinct(UserQuizScore.user_id)).label('today_unique_users')
    ).filter(
        func.date(UserQuizScore.completed_at) == today
    ).first()
    
    # Weekly growth
    week_ago = datetime.utcnow() - timedelta(days=7)
    last_week_stats = db.query(
        func.count(UserQuizScore.id).label('last_week_quizzes'),
        func.avg(UserQuizScore.score).label('last_week_avg_score')
    ).filter(
        UserQuizScore.completed_at >= week_ago - timedelta(days=7),
        UserQuizScore.completed_at < week_ago
    ).first()
    
    # Calculate growth percentages
    weekly_quiz_growth = 0
    weekly_score_growth = 0
    
    if last_week_stats and last_week_stats.last_week_quizzes > 0:
        weekly_quiz_growth = ((today_stats.today_quizzes or 0) - last_week_stats.last_week_quizzes) / last_week_stats.last_week_quizzes * 100
        weekly_score_growth = ((today_stats.today_avg_score or 0) - last_week_stats.last_week_avg_score) / last_week_stats.last_week_avg_score * 100
    
    return {
        "summary": {
            "total_quizzes_taken": total_quizzes_taken,
            "total_unique_users": total_unique_users,
            "average_score": round(average_score, 2),
            "total_questions_answered": total_questions_answered,
            "total_correct_answers": total_correct_answers,
            "overall_accuracy": round(overall_accuracy, 2),
            "today_quizzes": today_stats.today_quizzes or 0,
            "today_unique_users": today_stats.today_unique_users or 0,
            "today_avg_score": round(today_stats.today_avg_score or 0, 2)
        },
        "growth_metrics": {
            "weekly_quiz_growth": round(weekly_quiz_growth, 2),
            "weekly_score_growth": round(weekly_score_growth, 2),
            "trend": "up" if weekly_quiz_growth > 0 else "down"
        },
        "category_performance": [
            {
                "category_id": stat.id,
                "category_name": stat.name,
                "quiz_count": stat.quiz_count,
                "average_score": round(stat.avg_score or 0, 2),
                "unique_users": stat.unique_users
            }
            for stat in category_stats
        ],
        "daily_activity": [
            {
                "date": activity.date,
                "quiz_count": activity.quiz_count,
                "average_score": round(activity.avg_score or 0, 2),
                "unique_users": activity.unique_users
            }
            for activity in daily_activity
        ],
        "top_performers": [
            {
                "user_id": performer.user_id,
                "user_name": f"{performer.full_name} {performer.last_name}".strip(),
                "average_score": round(performer.avg_score or 0, 2),
                "quiz_count": performer.quiz_count,
                "best_score": round(performer.best_score or 0, 2),
                "performance_tier": get_performance_tier(performer.avg_score or 0)
            }
            for performer in top_performers
        ],
        "popular_categories": [
            {
                "category_id": category.id,
                "category_name": category.name,
                "attempt_count": category.attempt_count,
                "average_score": round(category.avg_score or 0, 2)
            }
            for category in popular_categories
        ],
        "difficulty_analysis": [
            {
                "difficulty": stat.difficulty,
                "quiz_count": stat.quiz_count,
                "average_score": round(stat.avg_score or 0, 2),
                "average_time_taken": round(stat.avg_time_taken or 0, 2)
            }
            for stat in difficulty_stats
        ]
    }