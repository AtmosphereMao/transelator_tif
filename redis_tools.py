import time,json,redis


class RedisTools(object):
    def __init__(self, host, port, pwd, db):
        self.r = redis.Redis(host=host, port=port, password=pwd, db=db, decode_responses=True)

    # 根据键值读字符串
    def get_values(self, key):
        dic = self.r.get(key)
        if dic:
            return json.loads(dic)
        else:
            return {}

    # 写入一个键值对
    def set_values(self, key, dic,ex=0):
        if ex == 0:
            self.r.set(key, json.dumps(dic))
        else:
            self.r.set(key, json.dumps(dic),ex)

    # 成功返回int 1
    def remove_key(self, key):
        return self.r.delete(key)

    # 有序集合
    def get_zrange(self, key, end = 0):
        return self.r.zrange(key, 0,end)
    
    def get_zrange_score(self, key, value):
        return self.r.zscore(key, value)

    def set_zrange(self, key, info):
        return self.r.zadd(key, info)

    def del_zrange(self, key, value):
        return self.r.zrem(key, value)

    def set_vaules_zrange(self, key, info):
        uuidStr = time.time()
        dic = {}
        dic = {uuidStr:uuidStr}
        self.set_values(key+":"+str(uuidStr),info)
        return self.r.zadd(key, dic)

    def set_list(self,key,value):
        return self.r.lpush(key, value)

    def get_list(self,key,end=-1):
        return self.r.lrange(key,0,end)

    def get_rpop_list(self,key):
        return self.r.rpop(key)

    def get_lrange_list(self,key,start=0,end=-1):
        return self.r.lrange(key,start,end)



    # 读队列中的第一个
    def read_queue(self, queue_name):
        range_list = self.r.zrange(queue_name, 0, 1)
        if range_list:
            set_del_task = range_list[0]
            try:
                set_del_task = json.loads(set_del_task)
            except:
                print("can not json") 
            return set_del_task

    # callback_obj字典对象
    def write_queue(self, queue_name, callback_obj):
        if type(callback_obj) != str:
            callback_obj = json.dumps(callback_obj)
            # print("no str")
        callback_mapping = {
            callback_obj: time.time()
        }
        self.r.zadd(queue_name, callback_mapping)

    def remove_queue(self, queue_name, callback_obj):
        if not self.r.zrem(queue_name, callback_obj):
            print("remove need json.dumps")
            try:    
                callback_obj = json.dumps(callback_obj)
            except:
                print("can not json")
            self.r.zrem(queue_name, callback_obj)


if __name__ == "__main__":
    r = RedisTools("localhost", 6379, "",0)

    # r.set_values("hello", {"hello":"world"})
    # r.write_queue("mytest","ok")
    # r.remove_key("hello")
    # r.remove_queue("mytest","ok")

    # print(r.get_values("hello"))
    # print(r.read_queue("mytest"))