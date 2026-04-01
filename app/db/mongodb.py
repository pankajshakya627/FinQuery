"""
MongoDB connection and utility functions.
"""
import motor.motor_asyncio
from app.core.config import get_settings
import structlog

logger = structlog.get_logger(__name__)
settings = get_settings()

class MongoDB:
    client: motor.motor_asyncio.AsyncIOMotorClient = None
    db = None

    @classmethod
    async def connect(cls):
        """Initialize MongoDB connection. Non-fatal — warns if unreachable."""
        if cls.client is not None:
            return

        try:
            cls.client = motor.motor_asyncio.AsyncIOMotorClient(
                settings.mongodb_url,
                serverSelectionTimeoutMS=5000,  # 5s timeout instead of 30s
            )
            db_name = settings.mongodb_url.split('/')[-1] if '/' in settings.mongodb_url else "hdfc_rag"
            cls.db = cls.client[db_name]

            # Verify connection with a short timeout
            await cls.client.admin.command('ping')
            logger.info("mongodb_connected", url=settings.mongodb_url)
        except Exception as e:
            # Non-fatal: warn but don't crash the app
            logger.warning("mongodb_unavailable_at_startup", error=str(e),
                           hint="Start MongoDB: brew services start mongodb-community@7.0")
            # Keep client=None so reconnect is retried on first request
            cls.client = None
            cls.db = None

    @classmethod
    async def disconnect(cls):
        """Close MongoDB connection."""
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None
            logger.info("mongodb_disconnected")

    @classmethod
    def get_db(cls):
        """Return the database instance."""
        return cls.db

async def get_mongodb():
    """Dependency for FastAPI to get MongoDB instance. Auto-reconnects if needed."""
    if MongoDB.client is None:
        await MongoDB.connect()
    return MongoDB.get_db()


async def check_mongodb_connection() -> bool:
    """Check if MongoDB is reachable."""
    try:
        if MongoDB.client is None:
            await MongoDB.connect()
        if MongoDB.client is None:
            return False
        await MongoDB.client.admin.command('ping')
        return True
    except Exception:
        return False
