import redis

def redis_connection():
    return redis.Redis(host="redis", port=6379)