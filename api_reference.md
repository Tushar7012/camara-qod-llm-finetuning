# CAMARA Quality on Demand (QoD) API Reference

## Overview

The Quality-On-Demand (QoD) API enables developers to request stable network performance (latency or throughput) without deep knowledge of underlying network complexity.

## Session Creation Endpoint

### POST /sessions

Creates a QoS session to request stable latency or throughput for application data flows.

**Request Body Structure:**

```json
{
 "device": {
 "ipv4Address": {
 "publicAddress": "string",
 "publicPort": 0
 },
 "ipv6Address": "string",
 "phoneNumber": "string"
 },
 "applicationServer": {
 "ipv4Address": "string",
 "ipv6Address": "string"
 },
 "qosProfile": "string",
 "devicePorts": {
 "ports": [0],
 "ranges": [
 {
 "from": 0,
 "to": 0
 }
 ]
 },
 "applicationServerPorts": {
 "ports": [0],
 "ranges": [
 {
 "from": 0,
 "to": 0
 }
 ]
 },
 "duration": 0,
 "sink": "string",
 "sinkCredential": {
 "credentialType": "ACCESSTOKEN",
 "accessToken": "string"
 }
}
```

### Required Fields

- **device**: At least ONE identifier (IPv4, IPv6, or phoneNumber/MSISDN)
- **applicationServer**: IPv4 and/or IPv6 address of backend
- **qosProfile**: Name of QoS profile (e.g., "QOS_E", "QOS_S", "QOS_M", "QOS_L")
- **duration**: Session duration in seconds

### Optional Fields

- **devicePorts**: Ports/ranges on device side
- **applicationServerPorts**: Ports/ranges on server side
- **sink**: Callback URL for notifications
- **sinkCredential**: Authentication for callback

## Device Identifiers

### IPv4 Address
```json
{
 "ipv4Address": {
 "publicAddress": "192.168.1.100",
 "publicPort": 5060
 }
}
```

### IPv6 Address
```json
{
 "ipv6Address": "2001:db8::1"
}
```

### Phone Number (MSISDN)
```json
{
 "phoneNumber": "+14155550123"
}
```

## QoS Profiles

### QOS_E (Enhanced)
- **Use Case**: Low-latency applications (gaming, VR, autonomous vehicles)
- **Characteristics**: 
 - Very low latency
 - Minimal jitter
 - High priority
 - Suitable for real-time interactive applications

### QOS_S (Streaming)
- **Use Case**: Live video streaming, video calls
- **Characteristics**:
 - Optimized for consistent throughput
 - Good balance of latency and bandwidth
 - Suitable for 4K/HD streaming

### QOS_M (Mission Critical)
- **Use Case**: IoT sensors, critical communications
- **Characteristics**:
 - Reliable delivery
 - Guaranteed minimum throughput
 - Suitable for industrial IoT

### QOS_L (Live)
- **Use Case**: Video conferencing, cloud gaming
- **Characteristics**:
 - Low latency for interactive media
 - Bi-directional optimization
 - Packet loss minimization

## Profile Parameters

Each QoS profile can include:

- **targetMinUpstreamRate**: Target minimum upload speed
- **maxUpstreamRate**: Maximum upload rate
- **maxUpstreamBurstRate**: Burst capacity for uploads
- **targetMinDownstreamRate**: Target minimum download speed
- **maxDownstreamRate**: Maximum download rate
- **maxDownstreamBurstRate**: Burst capacity for downloads
- **minDuration**: Minimum session duration allowed
- **maxDuration**: Maximum session duration allowed
- **priority**: Priority level (1-100, lower = higher priority)
- **packetDelayBudget**: Maximum one-way latency allowance
- **jitter**: Maximum variation in round-trip packet delay

## Example Session Requests

### Gaming Session
```json
{
 "device": {
 "ipv4Address": {
 "publicAddress": "203.0.113.45"
 }
 },
 "applicationServer": {
 "ipv4Address": "198.51.100.10"
 },
 "qosProfile": "QOS_E",
 "devicePorts": {
 "ports": [7777]
 },
 "applicationServerPorts": {
 "ports": [7777]
 },
 "duration": 3600
}
```

### 4K Video Streaming
```json
{
 "device": {
 "phoneNumber": "+14155551234"
 },
 "applicationServer": {
 "ipv4Address": "192.0.2.100"
 },
 "qosProfile": "QOS_S",
 "applicationServerPorts": {
 "ports": [443]
 },
 "duration": 7200
}
```

### IoT Sensor Upload
```json
{
 "device": {
 "ipv6Address": "2001:db8:85a3::8a2e:370:7334"
 },
 "applicationServer": {
 "ipv6Address": "2001:db8:1234::1"
 },
 "qosProfile": "QOS_M",
 "duration": 900
}
```

## Important Notes

1. **Device Identification**: Provide only ONE device identifier type
2. **Duration**: Must be within profile's min/max duration limits
3. **Ports**: Optional but recommended for precise flow identification
4. **Profile Selection**: Choose based on application requirements
5. **Session Lifecycle**: Sessions auto-terminate after duration expires

## Common Use Cases

| Scenario | QoS Profile | Typical Duration | Key Parameters |
|----------|-------------|------------------|----------------|
| Online Gaming | QOS_E | 1-4 hours | Low latency, minimal jitter |
| 4K Streaming | QOS_S | 2-3 hours | High downstream rate |
| Video Conference | QOS_L | 30-60 minutes | Bi-directional optimization |
| IoT Monitoring | QOS_M | 15-30 minutes | Reliable delivery |
| VR Application | QOS_E | 1-2 hours | Ultra-low latency |
| File Upload | QOS_S | Variable | High upstream rate |
