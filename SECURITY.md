# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Lifegence VoiceID SDK seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please send an email to:

**masakazu.nomura@lifegence.com**

Include the following information:

1. **Type of vulnerability** (e.g., authentication bypass, data exposure, injection)
2. **Location** of the affected source code (file path, line number if known)
3. **Step-by-step instructions** to reproduce the issue
4. **Proof of concept** or exploit code (if available)
5. **Impact assessment** of the vulnerability
6. **Suggested fix** (if you have one)

### What to Expect

| Timeline | Action |
|----------|--------|
| 24 hours | Initial acknowledgment of your report |
| 72 hours | Preliminary assessment and severity classification |
| 7 days | Detailed response with remediation plan |
| 90 days | Target for public disclosure (coordinated) |

### Our Commitment

- We will acknowledge receipt of your vulnerability report
- We will confirm the vulnerability and determine its impact
- We will release a fix as soon as reasonably possible
- We will credit you in the security advisory (unless you prefer anonymity)

## Security Best Practices for Users

### API Key Management

```bash
# DO: Use environment variables
export API_KEY="your-secure-api-key"

# DON'T: Hardcode in source code
API_KEY = "your-secure-api-key"  # Never do this!
```

### Production Deployment

1. **Change default credentials**
   ```yaml
   # docker-compose.yml - Change these for production!
   POSTGRES_PASSWORD: ${DB_PASSWORD}  # Use secrets management
   ```

2. **Enable HTTPS**
   - Always use TLS in production
   - Configure proper certificate management

3. **Restrict network access**
   - Use firewalls to limit API access
   - Consider VPN for internal services

4. **Monitor and audit**
   - Enable logging for all authentication attempts
   - Set up alerts for unusual patterns

### Biometric Data Protection

Voice embeddings are considered biometric data and require special handling:

| Requirement | Implementation |
|-------------|----------------|
| Encryption at rest | Use encrypted database storage |
| Encryption in transit | TLS 1.3 required |
| Access control | API key authentication |
| Data minimization | Store embeddings, not raw audio |
| Deletion capability | DELETE /speakers/{id} endpoint |
| Consent tracking | Required for enrollment |

## Known Security Considerations

### Voice Embedding Storage

- Embeddings are 192-dimensional vectors
- Cannot be reversed to reconstruct original voice
- Should still be treated as sensitive biometric data

### Replay Attack Mitigation

Current version does not include:
- Liveness detection
- Anti-spoofing measures

**Recommendation**: Implement additional verification factors for high-security applications.

### Rate Limiting

Default rate limits are configured but should be adjusted for production:

| Endpoint | Default Limit | Production Recommendation |
|----------|---------------|--------------------------|
| /enroll | 10/min | Reduce based on use case |
| /verify | 60/min | Adjust based on expected load |
| /identify | 30/min | Consider stricter limits |

## Security Updates

Security updates will be released as:

- **Critical**: Immediate patch release
- **High**: Patch within 7 days
- **Medium**: Next minor release
- **Low**: Next planned release

Subscribe to security notifications:
- Watch this repository for releases
- Check [CHANGELOG.md](./CHANGELOG.md) for security-related changes

## Compliance Considerations

This SDK is designed with the following regulations in mind:

### GDPR (EU)
- Article 9: Processing of special categories (biometric data)
- Article 17: Right to erasure (DELETE endpoint)
- Article 7: Consent requirements (enrollment consent tracking)

### BIPA (Illinois)
- Written consent before collection
- Purpose disclosure
- Retention and destruction policies

### CCPA (California)
- Right to know
- Right to delete
- Right to opt-out

**Note**: Compliance responsibility ultimately lies with the implementer. This SDK provides tools to support compliance but does not guarantee it.

## Acknowledgments

We appreciate the security research community's efforts in helping keep this project secure. Contributors who report valid vulnerabilities will be acknowledged here (with permission):

- *Your name could be here*

---

Thank you for helping keep Lifegence VoiceID SDK and its users safe!
