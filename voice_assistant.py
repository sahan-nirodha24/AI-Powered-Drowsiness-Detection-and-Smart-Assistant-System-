# voice_assistant.py

import os
import time
import pyttsx3

from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
load_dotenv()

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


class AIVoiceAssistant:

    def __init__(
        self,
        driver_name: Optional[str] = None,
        language: str = "en",
        use_cloud_assistant: bool = False,
        gemini_model_name: str = "models/gemini-2.5-flash",  # 2.5 
    ) -> None:

        self.driver_name = driver_name or "driver"
        self.language = language

        self.engine = pyttsx3.init("sapi5")
        self._configure_engine()

        self.use_cloud_assistant = use_cloud_assistant and GEMINI_AVAILABLE
        self.gemini_model_name = gemini_model_name
        self.gemini_model = None

        # Rate limit cooldown tracking
        self._gemini_blocked_until = 0.0

        # Throttle — calls  minimum 4s gap
        self._last_gemini_call = 0.0
        self.gemini_min_interval = 4.0

        # Cache —  command repeat  API call 
        self._gemini_cache: Dict[str, str] = {}
        self._gemini_cache_max = 30  # max cached entries

        if self.use_cloud_assistant:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("[VoiceAssistant] GEMINI_API_KEY not found. Cloud disabled.")
                self.use_cloud_assistant = False
            else:
                try:
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel(
                        model_name=self.gemini_model_name,
                        system_instruction=(
                            "You are a friendly AI driving assistant keeping a sleepy driver awake. "
                            "Use VERY simple, everyday English (max 2 sentences).\n"
                            "CRITICAL RULES:\n"
                            "1. If the driver asks you a question, ONLY answer it. STRICTLY DO NOT ask any questions back.\n"
                            "2. Only ask a follow-up question if the driver gives a short statement like 'yes' or 'no' to keep the conversation going.\n"
                            "3. Do NOT use complex words or hard trivia."
                        )
                    )
                    self.chat_session = self.gemini_model.start_chat(history=[])
                    print(f"[VoiceAssistant] Gemini enabled ({self.gemini_model_name}).")
                except Exception as e:
                    print(f"[VoiceAssistant] Failed to init Gemini: {e}")
                    self.use_cloud_assistant = False

    def _configure_engine(self) -> None:
        try:
            self.engine.setProperty("rate", 140)
            self.engine.setProperty("volume", 1.0)
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'zira' in voice.name.lower() or 'female' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        except Exception as e:
            print(f"[VoiceAssistant] TTS config error: {e}")

    def speak(self, text: str) -> None:
        if not text:
            return
        print(f"[Assistant] {text}")
        try:
            import subprocess, json, re
            # Clean LLM markdown symbols and emojis that make TTS sound garbled
            # Keep only alphanumeric characters, spaces, and basic punctuation (no double quotes to avoid backslash escaping issues in PowerShell)
            clean_text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", '', text)
            ps_text = json.dumps(clean_text)
            
            ps_script = (
                "Add-Type -AssemblyName System.Speech;"
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                "$s.Rate = -1; $s.Volume = 100;"
                # Select a clearer female voice (Zira) if available
                "$voice = $s.GetInstalledVoices() | Where-Object {$_.VoiceInfo.Name -match 'Zira'} | Select-Object -First 1;"
                "if ($voice) { $s.SelectVoice($voice.VoiceInfo.Name) };"
                f"$s.Speak({ps_text});"
            )
            
            cmd = ["powershell", "-Command", ps_script]
            # Use CREATE_NO_WINDOW if on Windows to prevent console flashing
            creationflags = 0x08000000 if os.name == 'nt' else 0
            subprocess.run(cmd, check=False, creationflags=creationflags)
        except Exception as e:
            print(f"[VoiceAssistant] TTS error: {e}")

    def build_drowsiness_message(self, level: str, context: Optional[Dict[str, Any]] = None) -> str:
        context = context or {}
        name = self.driver_name

        base = {
            "low":    f"{name}, I noticed early signs of tiredness.",
            "medium": f"{name}, you seem drowsy. Please slow down and consider taking a short break soon.",
            "high":   f"{name}, you are extremely drowsy. Please pull over at a safe place immediately and rest.",
        }

        msg = base.get(level.lower(), f"{name}, please stay alert.")

        if context.get("weather") in ["rainy", "foggy", "stormy"]:
            msg += " Road conditions are not ideal."

        if context.get("location_hint"):
            msg += f" There may be rest stops {context['location_hint']}."

        hour = datetime.now().hour
        if hour >= 23 or hour <= 5:
            msg += " It is late at night; fatigue risk is higher."

        return msg

    def alert_drowsiness(self, level: str, context: Optional[Dict[str, Any]] = None) -> None:
        msg = self.build_drowsiness_message(level, context)
        self.speak(msg)

    def handle_text_command(self, user_text: str) -> None:
        t = (user_text or "").lower().strip()

        if "exit" in t or "quit" in t:
            self.speak("Stopping the assistant. Drive safely.")
            raise SystemExit

        # Greetings
        if any(w in t for w in ["hello", "hi", "hey", "how are you", "how r u", "how are u", "wassup", "what's up", "whats up"]):
            self.speak(f"I'm here and ready to help, {self.driver_name}. Stay focused on the road.")

        # Time
        elif any(w in t for w in ["time", "clock", "what time"]):
            now = datetime.now().strftime("%H:%M")
            self.speak(f"The time is {now}.")

        # Drowsiness test
        elif "test" in t:
            self.alert_drowsiness("medium", {"speed": 60})

        # Driver feeling tired/sleepy
        elif any(w in t for w in ["tired", "sleepy", "drowsy", "sleep", "fatigue", "exhausted"]):
            self.speak(f"{self.driver_name}, please pull over safely and take a short rest. Your safety matters.")

        # Help request
        elif any(w in t for w in ["help", "assist", "support", "what can you do", "commands"]):
            self.speak("I can tell you the time, alert you about drowsiness, or just keep you company. Stay safe!")

        # Acknowledgements / filler
        elif any(w in t for w in ["ok", "okay", "thanks", "thank you", "got it", "alright", "sure", "fine"]):
            self.speak("Alright. Stay alert and drive safe.")

        # Fallback — friendly rather than an error message
        else:
            self.speak(f"I'm in offline mode right now, {self.driver_name}. I can help with the time or drowsiness alerts.")

    def _ask_cloud_assistant(self, user_text: str) -> Optional[str]:
        if not (self.use_cloud_assistant and self.gemini_model):
            return None

        # Rate limit cooldown check
        now = time.time()
        if now < self._gemini_blocked_until:
            remaining = int(self._gemini_blocked_until - now)
            print(f"[VoiceAssistant] Gemini cooldown: {remaining}s remaining. Using offline.")
            return None

        # Throttle — calls  minimum gap
        if now - self._last_gemini_call < self.gemini_min_interval:
            print("[VoiceAssistant] Throttled. Using offline.")
            return None

        # Bypassing cache for conversation mode to keep history dynamic
        
        try:
            response = self.chat_session.send_message(user_text)

            result = None
            if hasattr(response, "text"):
                result = response.text.strip()
            elif hasattr(response, "candidates") and response.candidates:
                parts = response.candidates[0].content.parts
                result = " ".join(getattr(p, "text", "") for p in parts).strip() or None

            if result:
                # Update throttle timestamp
                self._last_gemini_call = time.time()

            return result

        except Exception as e:
            error_str = str(e)

            # Rate limit detect  cooldown set 
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                # Parse a short retry delay (1–3 digits only) to avoid capturing
                # Unix timestamps or other large numbers in the error string.
                cooldown = 60  # default 60s
                try:
                    import re
                    # Only match 1–3 digit numbers (max 999s) before an "s" unit
                    match = re.search(r"retry[^0-9]*?(\d{1,3})\s*s\b", error_str, re.IGNORECASE)
                    if match:
                        parsed = int(match.group(1)) + 5  # add 5s buffer
                        # Sanity cap: never block for more than 5 minutes
                        cooldown = min(parsed, 300)
                except:
                    pass

                self._gemini_blocked_until = time.time() + cooldown
                print(f"[VoiceAssistant] Gemini rate limited. Offline mode for {cooldown}s.")
            else:
                print(f"[VoiceAssistant] Gemini error: {e}")

            return None

    def handle_command_with_nlp_backend(self, user_text: str) -> None:
        t = (user_text or "").lower().strip()

        if "exit" in t:
            self.handle_text_command(t)
            return

        # Drowsiness keywords - let Gemini handle it if cloud is enabled to start a conversation
        drowsy_keywords = ["tired", "sleepy", "drowsy", "sleep", "fatigue", "exhausted"]
        if not self.use_cloud_assistant and any(w in t for w in drowsy_keywords):
            self.handle_text_command(user_text)
            return

        if not self.use_cloud_assistant:
            self.handle_text_command(t)
            return

        # Gemini blocked → directly offline
        now = time.time()
        if now < self._gemini_blocked_until:
            self.handle_text_command(user_text)
            return

        # Try Gemini
        response = self._ask_cloud_assistant(user_text)

        if response:
            self.speak(response)
        else:
            self.handle_text_command(user_text)