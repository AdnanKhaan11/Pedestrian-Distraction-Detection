"""
Authentication service for user management and token handling.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
)
from app.db.models import User


class AuthService:
    """Service for authentication operations."""

    @staticmethod
    def create_user(
        db: Session,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None,
    ) -> User:
        """
        Create a new user.

        Args:
            db: Database session
            username: Username
            email: Email address
            password: Plain text password
            full_name: Full name (optional)

        Returns:
            Created user

        Raises:
            ValueError: If user already exists
        """
        # Check if user exists
        existing_user = (
            db.query(User)
            .filter((User.username == username) | (User.email == email))
            .first()
        )

        if existing_user:
            raise ValueError("Username or email already registered")

        # Create new user
        user = User(
            username=username,
            email=email,
            hashed_password=hash_password(password),
            full_name=full_name,
        )

        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """
        Authenticate user by username and password.

        Args:
            db: Database session
            username: Username
            password: Plain text password

        Returns:
            User if authenticated, None otherwise
        """
        user = db.query(User).filter(User.username == username).first()

        if not user or not verify_password(password, user.hashed_password):
            return None

        return user

    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """Get user by username."""
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def generate_tokens(user: User) -> dict:
        """
        Generate access and refresh tokens for user.

        Args:
            user: User object

        Returns:
            Dictionary with access and refresh tokens
        """
        access_token = create_access_token(
            data={"sub": str(user.id), "username": user.username}
        )
        refresh_token = create_refresh_token(
            data={"sub": str(user.id), "username": user.username}
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 30 * 60,  # 30 minutes in seconds
        }

    @staticmethod
    def change_password(
        db: Session, user_id: int, old_password: str, new_password: str
    ) -> bool:
        """
        Change user password.

        Args:
            db: Database session
            user_id: User ID
            old_password: Current password
            new_password: New password

        Returns:
            True if successful

        Raises:
            ValueError: If old password is incorrect
        """
        user = db.query(User).filter(User.id == user_id).first()

        if not user or not verify_password(old_password, user.hashed_password):
            raise ValueError("Current password is incorrect")

        user.hashed_password = hash_password(new_password)
        user.updated_at = datetime.utcnow()
        db.commit()
        return True
