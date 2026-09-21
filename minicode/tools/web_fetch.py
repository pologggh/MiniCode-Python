from __future__ import annotations

import urllib.request
import urllib.error
from minicode.tooling import ToolDefinition, ToolResult

MAX_CONTENT_LENGTH = 50000
MAX_REDIRECTS = 5  # 限制重定向次数防止 SSRF


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Check if URL is safe (not targeting internal, loopback, or private addresses via IP or DNS)."""
    try:
        import ipaddress
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(url)
        hostname = parsed.hostname

        if not hostname:
            return False, "Invalid URL: no hostname"

        hostname_lower = hostname.lower()
        if hostname_lower in {"localhost", "localhost.localdomain"}:
            return False, f"Access to localhost blocked: {hostname}"

        # Resolve hostname to IP addresses via DNS
        try:
            addr_info = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror as e:
            return False, f"DNS resolution failed for {hostname}: {e}"

        if not addr_info:
            return False, f"No IP addresses resolved for {hostname}"

        for entry in addr_info:
            sockaddr = entry[4]
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)

            if ip.is_loopback:
                return False, f"Loopback address blocked: {ip_str} ({hostname})"
            if ip.is_private:
                return False, f"Private network address blocked: {ip_str} ({hostname})"
            if ip.is_link_local:
                return False, f"Link-local address blocked: {ip_str} ({hostname})"
            if ip.is_multicast:
                return False, f"Multicast address blocked: {ip_str} ({hostname})"
            if ip.is_reserved:
                return False, f"Reserved address blocked: {ip_str} ({hostname})"
            if ip.is_unspecified:
                return False, f"Unspecified address blocked: {ip_str} ({hostname})"

        return True, "OK"
    except Exception as e:
        return False, f"URL validation failed: {e}"


def _validate(input_data: dict) -> dict:
    url = input_data.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("url is required")
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")
    max_chars = int(input_data.get("max_chars", 10000))
    if max_chars < 100 or max_chars > MAX_CONTENT_LENGTH:
        raise ValueError(f"max_chars must be between 100 and {MAX_CONTENT_LENGTH}")
    return {"url": url, "max_chars": max_chars}


def _run(input_data: dict, context) -> ToolResult:
    url = input_data["url"]
    max_chars = input_data["max_chars"]

    # SSRF Protection: verify target URL
    is_safe, reason = _is_safe_url(url)
    if not is_safe:
        return ToolResult(ok=False, output=f"Security Error: {reason}\nURL: {url}")

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "MiniCode-Python/0.5.0 (Terminal Coding Assistant)",
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            },
        )

        # 限制重定向次数并在重定向时再次校验目标地址
        class LimitedRedirectHandler(urllib.request.HTTPRedirectHandler):
            def __init__(self):
                self.redirect_count = 0
            
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                self.redirect_count += 1
                if self.redirect_count > MAX_REDIRECTS:
                    raise urllib.error.HTTPError(req.full_url, code, f"Too many redirects (>{MAX_REDIRECTS})", msg, fp)
                # Re-validate redirect target address
                is_redir_safe, redir_reason = _is_safe_url(newurl)
                if not is_redir_safe:
                    raise urllib.error.HTTPError(
                        req.full_url, 403, f"Redirect to unsafe address blocked: {redir_reason}", headers, fp
                    )
                return urllib.request.HTTPRedirectHandler.redirect_request(self, req, fp, code, msg, headers, newurl)


        opener = urllib.request.build_opener(LimitedRedirectHandler())
        
        with opener.open(req, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "")
            charset = "utf-8"

            if "charset=" in content_type:
                charset = content_type.split("charset=")[1].split(";")[0].strip()

            raw = response.read()
            try:
                text = raw.decode(charset, errors="replace")
            except (LookupError, UnicodeDecodeError):
                text = raw.decode("utf-8", errors="replace")

            # If HTML, try to extract meaningful content
            if "text/html" in content_type:
                text = _extract_text_from_html(text)

            # Truncate
            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars] + f"\n\n... [Content truncated at {max_chars} chars]"

            header = "\n".join([
                f"URL: {url}",
                f"CONTENT_TYPE: {content_type}",
                f"STATUS: {response.status}",
                f"CHARS: {len(text)}",
                f"TRUNCATED: {'yes' if truncated else 'no'}",
                "",
            ])

            return ToolResult(ok=True, output=header + text)

    except urllib.error.HTTPError as e:
        return ToolResult(
            ok=False,
            output=f"HTTP Error {e.code}: {e.reason}\nURL: {url}",
        )
    except urllib.error.URLError as e:
        return ToolResult(
            ok=False,
            output=f"Failed to fetch URL: {e.reason}\nURL: {url}",
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            output=f"Error fetching URL: {e}\nURL: {url}",
        )


def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML content."""
    import re

    # Remove script and style elements
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL)
    # Remove all tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


web_fetch_tool = ToolDefinition(
    name="web_fetch",
    description="Fetch content from a URL. Supports HTML (extracted to text), JSON, and plain text. Useful for reading documentation, APIs, or web content.",
    input_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch content from"},
            "max_chars": {"type": "number", "description": "Maximum characters to return (default: 10000)"},
        },
        "required": ["url"],
    },
    validator=_validate,
    run=_run,
)
