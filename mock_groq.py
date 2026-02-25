"""
mock_groq.py  –  Offline test stub for the groq SDK.

ONLY used by test files when the real 'groq' package is not installed.
Import via:
    try:
        import groq
    except ImportError:
        import mock_groq as groq
        import sys; sys.modules['groq'] = groq

Never place this file in a 'groq/' directory – that would shadow the
real SDK and break production runs.
"""

class APIError(Exception):
    def __init__(self, message="", response=None, body=None):
        super().__init__(message)
        self.response = response
        self.body     = body

class RateLimitError(APIError):
    pass

class AuthenticationError(APIError):
    pass

class BadRequestError(APIError):
    pass


class _Message:
    def __init__(self, content=""):
        self.content = content

class _Choice:
    def __init__(self, content=""):
        self.message = _Message(content)
        self.delta   = _Message(content)

class _Response:
    def __init__(self, content=""):
        self.choices = [_Choice(content)]

class _Completions:
    def create(self, **kwargs):
        return _Response("{}")

class _Chat:
    def __init__(self):
        self.completions = _Completions()

class Groq:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.chat    = _Chat()
