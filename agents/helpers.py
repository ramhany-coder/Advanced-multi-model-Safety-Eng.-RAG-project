import tempfile
from interfaces.Embedding import EmbeddingProvider



def tempfile_creator (audio_bytes,audio_formate):
    try : 
        temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=audio_formate
            )
        temp_file.write(audio_bytes)
        temp_file.flush()
        temp_file.close()
        audio_path = temp_file.name
    except Exception as e :
        raise ValueError(f"Error during creating temp file for audio scripting {e}")
    return temp_file , audio_path

def validate_router(self, router: str) -> str:
        if router not in self.routers_list:
            raise ValueError(
                f"Unsupported router '{router}'. "
                f"Available routers: {', '.join(self.routers_list)}"
            )
        return router

def clamp_text(text: str, max_chars: int = 2000, suffix: str = "...") -> str:
    """
    Clamps a string to a maximum number of characters.
    
    :param text: The input string to truncate.
    :param max_chars: Maximum allowed characters.
    :param suffix: String to append if truncated (default: '...').
    :return: Clamped string.
    """
    if not text:
        return ""
    
    text = str(text)
    
    if len(text) <= max_chars:
        return text
    
    # Adjust length to accommodate suffix
    cutoff = max(0, max_chars - len(suffix))
    return text[:cutoff] + suffix
