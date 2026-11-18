"""
IGPO Tool Server - Utilities (Web Search Only)
"""

import os
import json
import time
import uuid
import socket
import datetime
import threading
import traceback
from typing import List, Dict, Any


def string_to_uuid(input_string: str) -> str:
    """Convert string to deterministic UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(input_string)))


def get_network_info() -> str:
    """Get hostname for identification."""
    hostname = socket.gethostname()
    try:
        all_ips = socket.gethostbyname_ex(hostname)[2]
        real_ips = [ip for ip in all_ips if not ip.startswith("127.")]
        return hostname + (real_ips[0] if real_ips else "")
    except:
        return hostname


class FileSystemReader:
    """Simple local file system reader."""
    
    def __init__(self, **kwargs):
        pass
    
    def read_file(self, file_path: str) -> bytes:
        with open(file_path, 'rb') as f:
            return f.read()
    
    def write_file(self, file_path: str, content: Any, append: bool = False) -> bool:
        if isinstance(content, str):
            content = content.encode('utf-8')
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, 'ab' if append else 'wb') as f:
            f.write(content)
        return True
    
    def exists(self, file_path: str) -> bool:
        return os.path.exists(file_path) and not os.path.isdir(file_path)


class MessageClient:
    """Task submission client for web search."""
    
    def __init__(self, path: str = './cache/task_queue', **kwargs):
        self.path = path
        self._handler = None
    
    def submit_tasks(self, task_list: List[Dict]) -> List[Dict]:
        """Submit tasks and get results."""
        if not task_list:
            return task_list
        
        # Lazy init handler
        if self._handler is None:
            from tools_server.handler import Handler
            config = self._load_config()
            self._handler = Handler(config)
        
        try:
            return self._handler.handle_all(task_list)
        except Exception as e:
            print(f"[MessageClient] Error: {e}")
            traceback.print_exc()
            for task in task_list:
                if 'content' not in task:
                    task['content'] = f"Error: {str(e)}"
            return task_list
    
    def _load_config(self) -> Dict:
        """Load config from yaml."""
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {
                'search_engine': 'google',
                'search_top_k': 10,
                'cache_dir': './cache/tool_cache',
            }
