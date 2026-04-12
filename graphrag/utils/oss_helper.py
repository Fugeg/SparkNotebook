import os
import oss2
from dotenv import load_dotenv
import uuid
from datetime import datetime, timedelta

load_dotenv()

class OSSHelper:
    def __init__(self):
        self.access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
        self.access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET')
        self.bucket_name = os.getenv('OSS_BUCKET_NAME')
        self.endpoint = os.getenv('OSS_ENDPOINT')
        
        # 创建Bucket实例
        self.auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)
    
    def upload_file(self, local_file_path, remote_path=None):
        """
        上传本地文件到OSS
        
        Args:
            local_file_path: 本地文件路径
            remote_path: OSS上的路径，如果不指定则自动生成
            
        Returns:
            文件的公网访问URL
        """
        try:
            # 生成远程路径
            if remote_path is None:
                file_ext = os.path.splitext(local_file_path)[1]
                remote_path = f"audio/{datetime.now().strftime('%Y%m%d')}/{uuid.uuid4()}{file_ext}"
            
            # 上传文件
            self.bucket.put_object_from_file(remote_path, local_file_path)
            
            # 生成直接访问URL（不带签名）
            url = f'http://{self.bucket_name}.{self.endpoint}/{remote_path}'
            
            return url
        except Exception as e:
            print(f"上传文件到OSS失败: {e}")
            return None
    
    def delete_file(self, remote_path):
        """删除OSS上的文件"""
        try:
            self.bucket.delete_object(remote_path)
            return True
        except Exception as e:
            print(f"删除OSS文件失败: {e}")
            return False
    
    def delete_file_from_url(self, url):
        """从URL中提取路径并删除OSS文件"""
        try:
            # 从签名URL中提取对象路径
            # URL格式: https://bucket.endpoint/object?OSSAccessKeyId=...
            from urllib.parse import urlparse, unquote
            parsed = urlparse(url)
            # 路径格式: /bucket-name/object-key
            path = unquote(parsed.path)
            
            # 移除开头的斜杠和bucket名称
            if path.startswith('/'):
                path = path[1:]
            if path.startswith(self.bucket_name + '/'):
                path = path[len(self.bucket_name) + 1:]
            
            return self.delete_file(path)
        except Exception as e:
            print(f"从URL删除OSS文件失败: {e}")
            return False
