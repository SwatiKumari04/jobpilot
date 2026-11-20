"""All HTTP routes. Routes stay thin: validate input, call a service, shape the response."""
from . import admin, chat, digest, feedback, health, search, users

routers = [health.router, users.router, search.router, chat.router, digest.router, feedback.router, admin.router]

__all__ = ["routers"]
