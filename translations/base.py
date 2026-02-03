import json
import os
import subprocess
import time
from abc import ABC, abstractmethod

from tqdm import tqdm


class BaseTranslator(ABC):
    """
    Base translator class

    Responsibilities:
    - Start the appropriate llama.cpp container via docker-compose
      using the per-language MODEL_PATH.
    - Run a warm-up request.
    - Translate and write back to the diarization JSON.
    - Shut the container down gracefully.
    """

    # Must be overridden per language
    target_language = ""
    model_path = ""

    # docker compose settings
    service_name = "llama-server"
    project_name = "dubpls-translate"
    url = "http://127.0.0.1:8080/v1/chat/completions"
    
    compose_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "docker-compose.yaml"
    )

    # warm-up text
    warmup_text = "if you see this, translation is working!"

    @abstractmethod
    def translate_text(self, segment: dict) -> str:
        """
        Translate the provided text into the target language.
        """

    def translate_segments(self, json_path) -> None:
        """
        High-level orchestration:
        - start container
        - warm up
        - translate all segments
        - stop container
        """
        self._start_server()
        try:
            status = self._warm_up()
            if(not status):
                return

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for segment in tqdm(data.get("segments", [])):
                # src_text = segment.get("text", "")
                # word_count = len(src_text.split())
                # context = f"Keep the word count strictly between {word_count-1} and {word_count+1} words"
                translated = self.translate_text(segment)
                lang_key = self.target_language or "translation"
                segment[lang_key] = translated

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        finally:
            self._stop_server()

    def _compose_base_cmd(self) -> list[str]:
        return [
            "docker",
            "compose",
            "-f",
            self.compose_file,
            "-p",
            self.project_name,
        ]

    def _start_server(self) -> None:
        """
        Start the llama.cpp server container for this translator's model.
        """
        env = os.environ.copy()
        env["MODEL_PATH"] = self.model_path
        cmd = self._compose_base_cmd() + ["up", "-d", self.service_name]
        try:
            subprocess.run(cmd, check=True, env=env)
        except Exception as e:
            print(f"[E] Failed to start translation server: {e}")
            raise

        # grace period
        time.sleep(2)

    def _stop_server(self) -> None:
        """
        Stop the llama.cpp server container gracefully.
        """
        env = os.environ.copy()
        env["MODEL_PATH"] = self.model_path
        cmd = self._compose_base_cmd() + ["down"]
        try:
            subprocess.run(cmd, check=True, env=env)
        except Exception as e:
            print(f"[W] Failed to stop translation server cleanly: {e}")

    def _warm_up(self) -> None:
        """
        Runs the translation in target lang once, before bombarding
        the server with failing requests.
        """
        if not self.warmup_text:
            return
        print("warmup + test translation run")
        response = self.translate_text({"text": self.warmup_text})
        if response == "":
            print("Translation module isn't working as intended")
            self._print_docker_error()
            return False
        else:
            print(response)
            return True
    
    def _print_docker_error(self):
        cmd = [
            "docker",
            "container",
            "logs",
            f"{self.project_name}-{self.service_name}-1",
        ]
        subprocess.run(cmd, check=True, env=os.environ.copy())
