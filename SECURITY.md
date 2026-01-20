# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**DO NOT** file a public GitHub issue for security vulnerabilities.

Instead, please report security vulnerabilities by emailing:
security@edgefirst.ai

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Resolution Target**: Within 90 days (severity dependent)

## Disclosure Policy

We follow coordinated disclosure:
1. Reporter notifies us privately
2. We acknowledge and investigate
3. We develop and test a fix
4. We release the fix
5. We publicly disclose after users can update

## Security Updates

Security updates are released as patch versions and announced via:
- GitHub Security Advisories
- Release notes

## Scope

This library processes image data for training machine learning models. Security considerations include:

- **Input validation**: Image arrays are validated for expected shapes and types
- **Dependency security**: Dependencies are pinned to known-good versions
- **No network access**: This library does not make network requests
- **No file system writes**: This library only reads input data; it does not write files

## Best Practices for Users

- Keep dependencies updated to receive security patches
- Validate input data sources before processing
- Use virtual environments to isolate dependencies
