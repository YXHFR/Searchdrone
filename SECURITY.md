# Security policy

Searchdrone controls an aircraft. Security problems can become safety problems, so please report them privately.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting: open the repository's **Security** tab and choose **Report a vulnerability**. Do not open a public issue for security problems.

Please include:

- what is affected (package, file, version or commit),
- how to reproduce it,
- what an attacker could achieve, especially anything that could make the drone move, ignore STOP / RETURN HOME, or leave its approved area.

You will get an answer as soon as possible. This is a small open-source project without a bug bounty.

## Scope

In scope: everything in this repository, including the CI workflows and the review gate.

Especially important:

- any path where the AI commander could reach the flight controller without the safety gate,
- anything that weakens STOP / RETURN HOME, the geofence or the failsafes,
- leaks of secrets or personal data,
- services listening on anything other than localhost (or the LAN when explicitly enabled).

## Supported versions

Only the latest release and the `main` branch get fixes.
