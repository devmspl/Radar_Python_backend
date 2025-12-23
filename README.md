# 🎯 RADAR API - Backend

A powerful FastAPI-based backend application for the RADAR platform, providing comprehensive APIs for user management, content feeds, quizzes, bookmarks, YouTube integration, and more.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![Redis](https://img.shields.io/badge/Redis-5.0.1-red.svg)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0.23-orange.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)

---

## ✨ Features

| Feature                           | Description                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------------- |
| 🔐 **Authentication**             | Complete user registration, login, email verification, and password reset with OTP |
| 👤 **User Management**            | Profile management, photo uploads, admin controls                                  |
| 📚 **Content Feeds**              | Comprehensive feed management system                                               |
| 🎥 **YouTube Integration**        | YouTube API integration for video content                                          |
| 📝 **Quizzes**                    | Quiz creation and management                                                       |
| 🔖 **Bookmarks**                  | Save and organize content                                                          |
| 🔍 **Enhanced Search**            | Advanced search capabilities                                                       |
| 📰 **Web Scraping**               | Content scraping from various sources                                              |
| 📊 **Categories & Subcategories** | Organized content categorization                                                   |
| 🎓 **Onboarding**                 | User onboarding flow management                                                    |

---

## 🛠 Tech Stack

| Technology            | Purpose                                |
| --------------------- | -------------------------------------- |
| **FastAPI**           | Modern, high-performance web framework |
| **SQLAlchemy**        | ORM for database operations            |
| **Redis**             | OTP storage and caching                |
| **SQLite/PostgreSQL** | Database storage                       |
| **JWT**               | Secure authentication tokens           |
| **Pydantic**          | Data validation                        |
| **Uvicorn**           | ASGI server                            |
| **OpenAI**            | AI-powered features                    |
| **Selenium**          | Web scraping automation                |

---

## 📋 Prerequisites

Before running this project, ensure you have the following installed:

| Requirement | Version        |
| ----------- | -------------- |
| Python      | 3.10 or higher |
| Docker      | Latest         |
| pip         | Latest         |

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Radar_Python_backend
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\venv\Scripts\activate.bat

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Redis with Docker

Redis is required for OTP storage. Run the following command to start Redis:

```bash
docker run -d --name redis-server -p 6379:6379 redis:latest
```

**Useful Docker Commands for Redis:**

| Command                                       | Description                            |
| --------------------------------------------- | -------------------------------------- |
| `docker ps`                                   | Check if Redis is running              |
| `docker stop redis-server`                    | Stop Redis container                   |
| `docker start redis-server`                   | Start Redis container                  |
| `docker restart redis-server`                 | Restart Redis container                |
| `docker logs redis-server`                    | View Redis logs                        |
| `docker exec -it redis-server redis-cli ping` | Test Redis connection (returns `PONG`) |

---

## ⚙️ Configuration

Create a `.env` file in the root directory with the following environment variables:

```env
# Database
DATABASE_URL=sqlite:///./test.db

# Security
SECRET_KEY=your-super-secret-key-change-this-in-production

# SMTP Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Redis
REDIS_URL=redis://localhost:6379

# YouTube API
YOUTUBE_API_KEY=your-youtube-api-key

# OpenAI API
OPENAI_API_KEY=your-openai-api-key
```

### Environment Variables Reference

| Variable          | Description                         | Default                       |
| ----------------- | ----------------------------------- | ----------------------------- |
| `DATABASE_URL`    | Database connection string          | `sqlite:///./test.db`         |
| `SECRET_KEY`      | JWT secret key for token encryption | Required                      |
| `SMTP_SERVER`     | SMTP server for sending emails      | `smtp.gmail.com`              |
| `SMTP_PORT`       | SMTP server port                    | `587`                         |
| `SMTP_USERNAME`   | Email username for SMTP             | Required                      |
| `SMTP_PASSWORD`   | Email password/app password         | Required                      |
| `REDIS_URL`       | Redis connection URL                | `redis://localhost:6379`      |
| `YOUTUBE_API_KEY` | Google YouTube Data API key         | Required for YouTube features |
| `OPENAI_API_KEY`  | OpenAI API key                      | Required for AI features      |

---

## ▶️ Running the Application

### Development Mode (with auto-reload)

```bash
uvicorn main:app --port 8000 --reload
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Alternative (using Python directly)

```bash
python main.py
```

The server will start on `http://localhost:8000` (or port 7878 if using `python main.py`).

---

## 📖 API Documentation

Once the application is running, access the interactive API documentation:

| Documentation    | URL                                                                      |
| ---------------- | ------------------------------------------------------------------------ |
| **Swagger UI**   | [http://localhost:8000/docs](http://localhost:8000/docs)                 |
| **OpenAPI JSON** | [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json) |

### Authentication in Swagger UI

1. Click the **Authorize** button in Swagger UI
2. Enter your JWT token in the format: `your-jwt-token`
3. Click **Authorize** to authenticate all requests

---

## 📁 Project Structure

```
Radar_Python_backend/
├── main.py                 # Main application entry point & auth routes
├── config.py               # Application configuration & settings
├── database.py             # Database connection & session management
├── models.py               # SQLAlchemy ORM models
├── schemas.py              # Pydantic schemas for validation
├── dependencies.py         # FastAPI dependencies (auth, etc.)
├── utils.py                # Utility functions (OTP, email, hashing)
├── file_utils.py           # File handling utilities
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── test.db                 # SQLite database file
│
├── # Routers
├── admin.py                # Admin management routes
├── feed_router.py          # Content feed routes
├── search_router.py        # Search functionality routes
├── Enhanced_Search.py      # Enhanced search routes
├── youtube_router.py       # YouTube integration routes
├── youtube_service.py      # YouTube service logic
├── quiz_router.py          # Quiz management routes
├── bookmark_router.py      # Bookmark routes
├── publish_router.py       # Content publishing routes
├── onboarding_router.py    # User onboarding routes
├── Subcategory_router.py   # Subcategory management routes
├── addcategory.py          # Category management
│
├── # Utilities
├── scraping.py             # Web scraping routes
├── scrapper.py             # Scraping logic
│
└── uploads/                # Uploaded files directory
```

---

## 🔗 API Endpoints

### 🔐 Authentication (`/auth`)

| Method   | Endpoint                           | Description                |
| -------- | ---------------------------------- | -------------------------- |
| `POST`   | `/auth/register`                   | Register a new user        |
| `POST`   | `/auth/verify-email`               | Verify email with OTP      |
| `POST`   | `/auth/login`                      | User login                 |
| `POST`   | `/auth/forgot-password`            | Request password reset OTP |
| `POST`   | `/auth/verify-reset-otp`           | Verify password reset OTP  |
| `POST`   | `/auth/reset-password`             | Reset password             |
| `GET`    | `/auth/users/{user_id}`            | Get user by ID             |
| `GET`    | `/auth/users`                      | Get all users (Admin)      |
| `POST`   | `/auth/promote-to-admin/{user_id}` | Promote user to admin      |
| `PATCH`  | `/auth/users/{user_id}/block`      | Block/unblock user         |
| `DELETE` | `/auth/users/me`                   | Delete own account         |
| `DELETE` | `/auth/admin/users/{user_id}`      | Admin delete user          |
| `POST`   | `/auth/users/me/profile-photo`     | Upload profile photo       |
| `PUT`    | `/auth/users/me/profile-photo`     | Update profile photo       |
| `DELETE` | `/auth/users/me/profile-photo`     | Delete profile photo       |
| `PUT`    | `/auth/users/me/profile`           | Update user profile        |

### 📚 Other Routers

| Router              | Prefix             | Description                 |
| ------------------- | ------------------ | --------------------------- |
| **Admin**           | `/admin`           | Administrative operations   |
| **YouTube**         | `/youtube`         | YouTube content integration |
| **Feed**            | `/feed`            | Content feed management     |
| **Search**          | `/search`          | Search functionality        |
| **Enhanced Search** | `/enhanced-search` | Advanced search features    |
| **Publish**         | `/publish`         | Content publishing          |
| **Quiz**            | `/quiz`            | Quiz management             |
| **Onboarding**      | `/onboarding`      | User onboarding flow        |
| **Bookmark**        | `/bookmark`        | Bookmark management         |
| **Subcategory**     | `/subcategory`     | Subcategory management      |
| **Scraping**        | `/scrape`          | Web scraping operations     |

---

## 🔧 Troubleshooting

### Common Issues

| Issue                      | Solution                                                                            |
| -------------------------- | ----------------------------------------------------------------------------------- |
| **Redis connection error** | Ensure Redis is running: `docker ps` to check, `docker start redis-server` to start |
| **Email not sending**      | Verify SMTP credentials and enable "Less secure apps" or use App Password for Gmail |
| **Database errors**        | Delete `test.db` and restart the application to recreate tables                     |
| **Import errors**          | Ensure all dependencies are installed: `pip install -r requirements.txt`            |

### Test Redis Connection

```bash
docker exec -it redis-server redis-cli ping
# Should return: PONG
```

---

## 📝 License

This project is proprietary software. All rights reserved.

---

## 👨‍💻 Author

Developed by the RADAR Team

---

## 🤝 Contributing

Please contact the development team for contribution guidelines.
