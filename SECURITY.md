# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.14.x  | :white_check_mark: |
| 0.13.x  | :x:                |
| < 0.13  | :x:                |

## Reporting a Vulnerability

The vLLM-Omni team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings.

### Where to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing the vLLM security team or through the vLLM security channels:

- Email: [Contact vLLM team through their official channels](https://github.com/vllm-project/vllm)
- Slack: Join `#sig-omni` channel at [slack.vllm.ai](https://slack.vllm.ai) and contact maintainers privately

### What to Include

Please include the following information in your report:

- Type of vulnerability (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- We will acknowledge receipt of your vulnerability report within 3 business days
- We will send you a more detailed response within 7 days indicating the next steps
- We will keep you informed about the progress towards a fix and full announcement
- We may ask for additional information or guidance

## Security Best Practices

When using vLLM-Omni, please follow these security best practices:

1. **Keep Dependencies Updated**: Regularly update vLLM-Omni and its dependencies to get the latest security patches
2. **Validate Input**: Always validate and sanitize user inputs, especially when dealing with multimodal data
3. **Secure API Keys**: Never commit API keys or secrets to version control. Use environment variables or secure vaults
4. **Network Security**: When deploying vLLM-Omni servers, ensure they are behind proper authentication and network security measures
5. **Model Security**: Be cautious when loading models from untrusted sources. Models can contain malicious code in custom layers

## Known Security Considerations

### Pickle Usage
vLLM-Omni uses pickle for model serialization in specific cases. Pickle can execute arbitrary code during deserialization. We:
- Maintain a whitelist of allowed pickle imports
- Use pre-commit hooks to prevent unauthorized pickle usage
- Recommend only loading models from trusted sources

### Multimodal Input Processing
When processing images, audio, and video:
- Validate file formats and sizes
- Set resource limits to prevent DoS attacks
- Sanitize metadata that could contain malicious payloads

## Security Updates

Security updates will be announced through:
- GitHub Security Advisories
- Release notes in [Releases](https://github.com/vllm-project/vllm-omni/releases)
- vLLM community channels (Slack, Forum)

## Attribution

We appreciate security researchers who help keep vLLM-Omni and our community safe. With your permission, we will acknowledge your contribution in our security advisories.
