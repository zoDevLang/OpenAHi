# Security Policy

## Overview

OpenAHI takes security seriously. This document outlines our security policy and provides guidelines for reporting security vulnerabilities.

## Supported Versions

Security updates are provided for the following versions:

| Version | Supported |
|---------|----------|
| 0.1.x   | ✅ Yes   |
| < 0.1.0 | ❌ No    |

## Reporting a Vulnerability

If you discover a security vulnerability in OpenAHI, please follow these steps:

1. **Do not** open a public issue on GitHub
2. **Do not** discuss the vulnerability in public forums
3. **Do** report the vulnerability privately via email or GitHub security advisory

### Private Reporting

Please report security vulnerabilities by:

1. **Email**: Send an email to the maintainers with details of the vulnerability
2. **GitHub Security Advisory**: Create a private security advisory on GitHub

Include the following information in your report:

- **Type of vulnerability** (e.g., remote code execution, denial of service, etc.)
- **Steps to reproduce** the vulnerability
- **Impact** of the vulnerability
- **Suggested fix** or mitigation (if you have one)
- **Your contact information** (optional)

### What to Expect

1. **Acknowledgment**: You will receive an acknowledgment of your report within 48 hours
2. **Assessment**: We will assess the vulnerability and determine its severity
3. **Fix**: We will work on a fix for the vulnerability
4. **Disclosure**: We will coordinate with you on public disclosure

## Security Best Practices

### For Users

1. **Download from trusted sources**: Only download models and files from trusted sources
2. **Verify checksums**: Always verify file checksums when downloading
3. **Sandbox execution**: Consider running untrusted models in a sandboxed environment
4. **Keep updated**: Use the latest version of OpenAHI to get security fixes
5. **Review permissions**: Be careful with file permissions when running OpenAHI

### For Developers

1. **Input validation**: Always validate user input
2. **Sanitize file paths**: Prevent path traversal attacks
3. **Checksum verification**: Verify checksums of downloaded files
4. **Sandbox untrusted code**: Never execute untrusted code with full permissions
5. **Dependency security**: Keep dependencies updated and audit them for vulnerabilities

## Known Security Considerations

### Model Files

- Model files can be large and may contain arbitrary data
- Always verify checksums of downloaded model files
- Be cautious when loading model files from untrusted sources

### Inference

- Inference can consume significant computational resources
- Implement rate limiting to prevent resource exhaustion
- Be careful with user-provided prompts (prompt injection attacks)

### Training

- Training consumes significant computational resources
- Training data should be from trusted sources
- Be careful with custom training scripts

## Security Features

OpenAHI includes the following security features:

1. **Checksum Verification**: Model files are verified against checksums
2. **Sandboxing Support**: Runtime supports sandboxed execution (when enabled)
3. **Input Validation**: All inputs are validated before processing
4. **Resource Limits**: Configurable limits on memory and computation
5. **Secure Downloads**: HTTPS-only downloads with certificate verification

## Security Advisories

Past security advisories will be published in the [GitHub Security Advisories](https://github.com/zoDevLang/OpenAHi/security/advisories) section.

## Contact

For security-related questions, please contact the maintainers privately.

---

*Last updated: January 2024*
