"""
RBAC Authentication module.

Roles:
    "user"  — self-registered, can predict and view own history
    "admin" — can manage users, view all predictions, trigger retrain

Uses bcrypt for password hashing.
Streamlit session_state tracks the authenticated user.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import bcrypt
import streamlit as st
from sqlalchemy.exc import IntegrityError

from f1_predictor.common.database import User, get_db, init_db
from f1_predictor.common.exceptions import (
    AccountDisabledError,
    InsufficientPermissionsError,
    InvalidCredentialsError,
    UserAlreadyExistsError,
)
from f1_predictor.common.logger import get_logger

log = get_logger(__name__)

_SESSION_USER     = "auth_username"
_SESSION_ROLE     = "auth_role"
_SESSION_USER_ID  = "auth_user_id"


# ── Password helpers ──────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── Registration ──────────────────────────────────────────────────────────────

def register_user(username: str, email: str, password: str, role: str = "user") -> User:
    """
    Create a new user account.

    Args:
        username: Unique login name (3–30 chars).
        email:    Unique email address.
        password: Plain-text password (hashed before storage).
        role:     "user" (default) or "admin".

    Raises:
        UserAlreadyExistsError: If username or email is already taken.
        ValueError: If inputs fail basic validation.
    """
    username = username.strip().lower()
    email    = email.strip().lower()

    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    if "@" not in email:
        raise ValueError("Invalid email address.")

    init_db()

    with get_db() as db:
        existing = (
            db.query(User)
            .filter((User.username == username) | (User.email == email))
            .first()
        )
        if existing:
            raise UserAlreadyExistsError(
                f"Username '{username}' or email is already registered."
            )

        user = User(
            username=username,
            email=email,
            password_hash=_hash_password(password),
            role=role,
            is_active=True,
            created_at=datetime.utcnow(),
        )
        db.add(user)
        db.flush()
        db.refresh(user)
        user_id = user.id

    log.info(f"New user registered: {username!r} (role={role})")
    return user_id


# ── Login / Logout ────────────────────────────────────────────────────────────

def login(username: str, password: str) -> dict:
    """
    Authenticate and populate Streamlit session state.

    Returns:
        dict with keys: username, role, user_id

    Raises:
        InvalidCredentialsError: Wrong username or password.
        AccountDisabledError: Account exists but is deactivated.
    """
    username = username.strip().lower()
    init_db()

    with get_db() as db:
        user = db.query(User).filter_by(username=username).first()

        if user is None or not _verify_password(password, user.password_hash):
            raise InvalidCredentialsError("Incorrect username or password.")

        if not user.is_active:
            raise AccountDisabledError(
                "This account has been deactivated. Contact an administrator."
            )

        user.last_login = datetime.utcnow()
        db.flush()

        st.session_state[_SESSION_USER]    = user.username
        st.session_state[_SESSION_ROLE]    = user.role
        st.session_state[_SESSION_USER_ID] = user.id

    log.info(f"User logged in: {username!r}")
    return {
        "username": st.session_state[_SESSION_USER],
        "role":     st.session_state[_SESSION_ROLE],
        "user_id":  st.session_state[_SESSION_USER_ID],
    }


def logout() -> None:
    """Clear auth session state."""
    for key in [_SESSION_USER, _SESSION_ROLE, _SESSION_USER_ID]:
        st.session_state.pop(key, None)
    log.info("User logged out.")


# ── Session helpers ───────────────────────────────────────────────────────────

def current_user() -> Optional[str]:
    return st.session_state.get(_SESSION_USER)


def current_role() -> Optional[str]:
    return st.session_state.get(_SESSION_ROLE)


def current_user_id() -> Optional[int]:
    return st.session_state.get(_SESSION_USER_ID)


def is_authenticated() -> bool:
    return _SESSION_USER in st.session_state


def is_admin() -> bool:
    return st.session_state.get(_SESSION_ROLE) == "admin"


def require_auth() -> str:
    """
    Guard: redirect to login if not authenticated.
    Returns the current username if authenticated.
    """
    if not is_authenticated():
        st.stop()   # Caller (app.py) renders login before calling this
    return current_user()


def require_role(role: str) -> None:
    """Guard: stop execution if the user's role is insufficient."""
    role_rank = {"user": 1, "admin": 2}
    current = current_role() or "user"
    if role_rank.get(current, 0) < role_rank.get(role, 99):
        raise InsufficientPermissionsError(
            f"This action requires role '{role}'. Your role: '{current}'."
        )


# ── Streamlit UI Components ───────────────────────────────────────────────────

def render_login_page() -> bool:
    """
    Render the full login/register page.
    Returns True if the user successfully authenticated this call.
    """
    st.markdown(
        """
        <style>
        .auth-container { max-width: 420px; margin: auto; padding-top: 60px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.title("🏎️ F1 Prediction Center")
    st.markdown("#### Sign in to access predictions and analytics")
    st.markdown("---")

    tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

    with tab_login:
        with st.form("login_form"):
            uname = st.text_input("Username")
            pwd   = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if not uname or not pwd:
                st.error("Please enter both username and password.")
            else:
                try:
                    login(uname, pwd)
                    st.success(f"Welcome back, {current_user()}! 🏁")
                    st.rerun()
                    return True
                except (InvalidCredentialsError, AccountDisabledError) as exc:
                    st.error(str(exc))

    with tab_register:
        with st.form("register_form"):
            new_user  = st.text_input("Choose a username")
            new_email = st.text_input("Email address")
            new_pwd   = st.text_input("Password (min 6 chars)", type="password")
            new_pwd2  = st.text_input("Confirm password", type="password")
            reg_btn   = st.form_submit_button("Create Account", use_container_width=True)

        if reg_btn:
            if new_pwd != new_pwd2:
                st.error("Passwords do not match.")
            else:
                try:
                    register_user(new_user, new_email, new_pwd)
                    st.success("Account created! Please log in.")
                except (UserAlreadyExistsError, ValueError) as exc:
                    st.error(str(exc))

    st.markdown('</div>', unsafe_allow_html=True)
    return False


def render_admin_panel() -> None:
    """
    Full admin control panel — visible only to admins.
    Shows: all users, toggle active, promote to admin, model run log.
    """
    require_role("admin")
    st.subheader("👑 Admin Panel")

    with get_db() as db:
        users = db.query(User).order_by(User.created_at.desc()).all()
        user_data = [
            {
                "ID": u.id,
                "Username": u.username,
                "Email": u.email,
                "Role": u.role,
                "Active": u.is_active,
                "Created": u.created_at.strftime("%Y-%m-%d") if u.created_at else "",
                "Last Login": u.last_login.strftime("%Y-%m-%d %H:%M") if u.last_login else "Never",
            }
            for u in users
        ]

    import pandas as pd
    df_users = pd.DataFrame(user_data)
    st.dataframe(df_users, use_container_width=True, hide_index=True)

    st.markdown("#### Manage User")
    col1, col2, col3 = st.columns(3)

    with col1:
        target = st.text_input("Username to manage")
    with col2:
        action = st.selectbox("Action", ["Deactivate", "Activate", "Promote to Admin", "Demote to User"])
    with col3:
        st.write("")
        st.write("")
        if st.button("Apply", use_container_width=True):
            _apply_admin_action(target.strip().lower(), action)


def _apply_admin_action(username: str, action: str) -> None:
    """Execute an admin action on a target user."""
    if username == current_user():
        st.error("You cannot modify your own account.")
        return
    with get_db() as db:
        user = db.query(User).filter_by(username=username).first()
        if user is None:
            st.error(f"User '{username}' not found.")
            return
        if action == "Deactivate":
            user.is_active = False
            st.success(f"'{username}' deactivated.")
        elif action == "Activate":
            user.is_active = True
            st.success(f"'{username}' activated.")
        elif action == "Promote to Admin":
            user.role = "admin"
            st.success(f"'{username}' promoted to admin.")
        elif action == "Demote to User":
            user.role = "user"
            st.success(f"'{username}' demoted to user.")
    log.info(f"Admin {current_user()!r} applied '{action}' to '{username}'")
