import time
import queue
import logging
from jupyter_client import KernelManager

logger = logging.getLogger(__name__)

class LocalIPythonRunner:
    """Manages a persistent local IPython kernel for stateful execution."""
    
    def __init__(self):
        self.km = KernelManager()
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        # Wait for kernel to be ready
        self.kc.wait_for_ready(timeout=10)
        logger.info("Persistent IPython kernel started.")

    def execute(self, code: str, timeout: int = 300) -> dict:
        """Executes code in the persistent kernel."""
        # Clear stdout/stderr queue
        msg_id = self.kc.execute(code)
        
        output = []
        error = None
        
        start = time.time()
        while (time.time() - start) < timeout:
            try:
                # Poll messages
                msg = self.kc.get_iopub_msg(timeout=1)
                msg_type = msg['header']['msg_type']
                content = msg.get('content', {})
                
                if msg_type == 'stream':
                    output.append(content.get('text', ''))
                elif msg_type == 'execute_result' or msg_type == 'display_data':
                    output.append(content.get('data', {}).get('text/plain', ''))
                elif msg_type == 'error':
                    error = f"{content.get('ename')}: {content.get('evalue')}"
                    break
                elif msg_type == 'status' and content.get('execution_state') == 'idle':
                    # Kernel finished
                    break
            except queue.Empty:
                continue
                
        return {
            "success": error is None,
            "output": "".join(output),
            "error": error,
        }

    def shutdown(self):
        self.kc.stop_channels()
        self.km.shutdown_kernel()
        logger.info("Kernel shut down.")
