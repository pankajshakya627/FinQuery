import motor.motor_asyncio
import structlog
from app.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

class MongoDB:
    client: motor.motor_asyncio.AsyncIOMotorClient = None
    db = None
    @classmethod
    def get_db(cls):
        return cls.db

    @classmethod
    async def connect(cls):
        if cls.client is not None:
            return
        
        try:
            logger.info("connecting_mongodb", url=settings.mongodb_url)
            cls.client = motor.motor_asyncio.AsyncIOMotorClient(
                settings.mongodb_url,
                serverSelectionTimeoutMS=5000,
            )
            db_name = settings.mongodb_url.split('/')[-1] if '/' in settings.mongodb_url else "finquery_rag"
            cls.db = cls.client[db_name]

            # Verify connection
            await cls.client.admin.command('ping')
            logger.info("mongodb_connected", db=db_name)
        except Exception as e:
            logger.error("mongodb_connection_failed", error=str(e))
            cls.client = None
            cls.db = None
            raise

    @classmethod
    async def disconnect(cls):
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None
            logger.info("mongodb_connection_closed")

async def get_mongodb():
    if MongoDB.db is None:
        await MongoDB.connect()
    return MongoDB.db

async def check_mongodb_connection():
    try:
        if MongoDB.client is None:
            await MongoDB.connect()
        if MongoDB.client is None:
            return False
        await MongoDB.client.admin.command('ping')
        return True
    except Exception:
        return False
