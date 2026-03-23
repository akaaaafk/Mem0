import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


# Update custom instructions
custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


class MemoryADD:
    def __init__(self, data_path=None, batch_size=2, mem0_client=None):
        """LoCoMo 仅 vector-only，不开图。mem0_client 可选，用于 add_then_search 复用同一连接避免 Qdrant 锁冲突。"""
        if mem0_client is not None:
            self.mem0_client = mem0_client
            self._use_local = not hasattr(mem0_client, "update_project")
        else:
            api_key = os.getenv("MEM0_API_KEY")
            if api_key:
                from mem0 import MemoryClient
                self.mem0_client = MemoryClient(
                    api_key=api_key,
                    org_id=os.getenv("MEM0_ORGANIZATION_ID"),
                    project_id=os.getenv("MEM0_PROJECT_ID"),
                )
                self.mem0_client.update_project(custom_instructions=custom_instructions)
                self._use_local = False
            else:
                from src.memzero.local_memory import create_local_memory
                self.mem0_client = create_local_memory()
                self._use_local = True

        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        for attempt in range(retries):
            try:
                if self._use_local:
                    _ = self.mem0_client.add(message, user_id=user_id, metadata=metadata)
                else:
                    _ = self.mem0_client.add(message, user_id=user_id, version="v2", metadata=metadata)
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # delete all memories for the two users
        # 仅在使用远程 Mem0 API 时清空；本地 Qdrant 后端在本次评估中会为每个 run 使用独立 MEM0_DIR，
        # 且 user_id 已带上 idx，不会与其它会话冲突，避免多线程下对本地 Qdrant 频繁 reset 导致的 Collection not found。
        if not self._use_local:
            self.mem0_client.delete_all(user_id=speaker_a_user_id)
            self.mem0_client.delete_all(user_id=speaker_b_user_id)

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            messages = []
            messages_reverse = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            # add memories for the two users on different threads
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A"),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B"),
            )

            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

        print("Messages added successfully")

    def process_all_conversations(self, max_workers=10, max_samples=None, max_questions=None):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        if max_questions is not None:
            # Process as many conversations as needed to cover at least max_questions questions
            n = 0
            total_q = 0
            for item in self.data:
                total_q += len(item.get("qa", []))
                n += 1
                if total_q >= max_questions:
                    break
            data = self.data[:n]
        elif max_samples is not None:
            data = self.data[:max_samples]
        else:
            data = self.data
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(data)]

            for future in futures:
                future.result()
