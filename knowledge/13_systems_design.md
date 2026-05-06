# Distributed Computing and System Design

## Microservices Architecture

### Service Decomposition

Microservices decompose a monolithic application into independently deployable services organized around business capabilities. Each service owns its data, communicates through well-defined APIs, and can be scaled independently. The key principle is that services should be small enough to be owned by a single team (the "two-pizza team" rule) but large enough to be independently meaningful.

Service boundaries should follow the Bounded Context pattern from Domain-Driven Design (DDD). A bounded context defines a boundary within which a particular domain model is consistent. For example, a "User" in the authentication context (credentials, sessions) is different from a "User" in the billing context (payment methods, invoices).

Anti-patterns in service decomposition include:
- **Distributed monolith**: Services are tightly coupled through shared databases or synchronous call chains. Changes to one service require coordinated deployment of multiple services.
- **Nano-services**: Services are too fine-grained, leading to excessive network overhead and operational complexity.
- **Shared database**: Multiple services access the same database tables, creating implicit coupling.

### Inter-Service Communication

**Synchronous communication** (REST, gRPC) is appropriate when the caller needs an immediate response. REST uses HTTP/JSON for simplicity and interoperability. gRPC uses Protocol Buffers for efficient binary serialization and supports bidirectional streaming.

**Asynchronous communication** (message queues, event streaming) decouples services temporally. Apache Kafka provides durable, ordered, replayable event streams. RabbitMQ provides traditional message queue semantics with routing, dead-letter queues, and acknowledgments.

The Saga pattern coordinates distributed transactions across services. Each service performs its local transaction and publishes an event. If any step fails, compensating transactions undo the effects of previous steps. Choreography-based sagas use events (decentralized). Orchestration-based sagas use a central coordinator.

### API Gateway and Service Mesh

An API gateway provides a single entry point for external clients, handling cross-cutting concerns:
- **Routing**: Forward requests to appropriate backend services
- **Rate limiting**: Protect services from traffic spikes
- **Authentication**: Validate tokens before forwarding
- **Response aggregation**: Combine responses from multiple services
- **Protocol translation**: REST to gRPC conversion

A service mesh (Istio, Linkerd) handles inter-service communication concerns at the infrastructure layer:
- **Traffic management**: Load balancing, circuit breaking, retries, timeouts
- **Security**: Mutual TLS between services, authorization policies
- **Observability**: Automatic request tracing, metrics collection

## Containerization and Orchestration

### Docker Architecture

Docker containers package applications with their dependencies into lightweight, portable units. The layered filesystem (Union File System) enables efficient image storage and distribution. Each instruction in a Dockerfile creates a new layer; unchanged layers are cached and shared between images.

Multi-stage builds reduce image size by separating build and runtime environments. The build stage includes compilers and development tools; the runtime stage copies only the built artifacts. This can reduce image sizes from 1GB+ to <100MB.

Docker Compose defines multi-container applications in a declarative YAML file. Services can depend on each other, share networks, and mount volumes. Health checks enable dependency-aware startup ordering. For development environments, Compose provides hot-reload through volume mounts.

### Kubernetes Orchestration

Kubernetes automates deployment, scaling, and management of containerized applications. The core abstractions are:

- **Pod**: Smallest deployable unit containing one or more containers sharing network and storage
- **Deployment**: Manages a set of Pod replicas with declarative updates and rollbacks
- **Service**: Stable network endpoint for a set of Pods (ClusterIP, NodePort, LoadBalancer)
- **Ingress**: HTTP/HTTPS routing rules for external access
- **ConfigMap/Secret**: Configuration and sensitive data management
- **PersistentVolumeClaim**: Storage requests that abstract underlying storage providers

Horizontal Pod Autoscaler (HPA) adjusts replica count based on CPU, memory, or custom metrics. Vertical Pod Autoscaler (VPA) adjusts resource requests and limits. Cluster Autoscaler adds or removes nodes based on pending pod scheduling.

## Stream Processing

### Apache Kafka Architecture

Kafka organizes messages into topics, each divided into partitions for parallelism. Producers append messages to partition logs; consumers read sequentially from an offset. Consumer groups distribute partitions among group members for parallel consumption.

Key guarantees:
- **Ordering**: Messages within a partition are strictly ordered
- **Durability**: Messages are replicated across brokers with configurable acknowledgment
- **Exactly-once semantics**: Achieved through idempotent producers and transactional APIs

Kafka Streams processes data in real-time using a library (not a separate cluster). It supports stateful operations (aggregation, windowing, joins) with automatic state management backed by RocksDB.

### Event Sourcing and CQRS

Event Sourcing stores state changes as a sequence of immutable events rather than mutable current state. The current state is derived by replaying events from the beginning. This provides a complete audit trail, enables temporal queries, and supports multiple read models.

Command Query Responsibility Segregation (CQRS) separates the write model (optimized for updates) from the read model (optimized for queries). Combined with Event Sourcing, the write side appends events and the read side projects events into denormalized views optimized for specific query patterns.

## Reliability Engineering

### Site Reliability Engineering (SRE) Principles

SRE defines reliability targets through Service Level Objectives (SLOs):
- **SLI (Service Level Indicator)**: Quantitative measure (request latency, error rate, throughput)
- **SLO (Service Level Objective)**: Target value for an SLI (99.9% of requests under 200ms)
- **SLA (Service Level Agreement)**: Business contract with consequences for missing SLOs

Error budgets represent the acceptable unreliability: if the SLO is 99.9%, the error budget is 0.1%. When error budget is consumed, the team prioritizes reliability over features. This creates a quantitative framework for balancing velocity and stability.

### Fault Tolerance Patterns

**Circuit Breaker**: Monitors call failures and "opens" when the failure rate exceeds a threshold, immediately returning errors without attempting calls. After a timeout, allows a single test call ("half-open"). If it succeeds, the circuit "closes" and normal operation resumes.

**Bulkhead**: Isolates components so that failure in one doesn't cascade to others. Thread pool bulkheads allocate separate thread pools for different dependencies. Connection pool bulkheads limit connections per downstream service.

**Retry with Exponential Backoff**: Retries failed requests with increasing delays (1s, 2s, 4s, 8s...) plus jitter (random offset to prevent thundering herd). Maximum retry count and maximum backoff prevent infinite loops.

**Timeout**: Set explicit timeouts for all external calls. Distinguish between connection timeout (time to establish connection) and read timeout (time to receive response). Timeouts should be shorter than user-facing latency budgets.

## Security

### Authentication and Authorization

**OAuth 2.0** separates authentication (who are you?) from authorization (what can you do?). The authorization server issues access tokens after the user authenticates. Resource servers validate tokens before serving requests.

**JWT (JSON Web Tokens)** encode claims in a signed, base64-encoded token. Claims include issuer, subject, audience, expiration, and custom data. HMAC (HS256) uses symmetric keys; RSA (RS256) uses asymmetric keys for verification without the signing key.

**RBAC (Role-Based Access Control)** assigns permissions to roles, and roles to users. Hierarchical roles (admin > editor > viewer) simplify permission management. Attribute-Based Access Control (ABAC) evaluates policies based on user attributes, resource attributes, and environmental conditions.

### API Security

Rate limiting prevents abuse and ensures fair resource allocation. Algorithms include:
- **Token bucket**: Tokens are added at a fixed rate; each request consumes a token. Allows burst traffic up to bucket capacity.
- **Sliding window**: Counts requests in a sliding time window. More precise than fixed window, avoids boundary issues.
- **Leaky bucket**: Processes requests at a fixed rate, queuing excess requests.

Input validation prevents injection attacks:
- SQL injection: Use parameterized queries, never string concatenation
- XSS: Sanitize and escape user input in HTML output
- SSRF: Validate and whitelist external URLs
- Path traversal: Normalize paths and validate against allowed directories

## Performance Optimization

### Database Query Optimization

Query optimization strategies:
1. **Indexing**: Create indexes on columns used in WHERE, JOIN, and ORDER BY clauses. Composite indexes should match query column order (leftmost prefix rule).
2. **Query rewriting**: Replace correlated subqueries with JOINs. Use EXISTS instead of IN for existence checks. Avoid SELECT *; list specific columns.
3. **Partitioning**: Divide large tables into smaller partitions based on date, region, or hash. Partition pruning skips irrelevant partitions during queries.
4. **Materialized views**: Pre-compute expensive aggregations. Refresh periodically or incrementally.
5. **Connection management**: Use connection pooling, prepared statements, and batch operations.

### Caching Architecture

Multi-level caching hierarchy:
- **L1 (In-process)**: LRU cache in application memory. Fastest access (~1μs) but limited size and not shared between instances.
- **L2 (Distributed)**: Redis or Memcached. Network access (~1ms) but shared across instances and survives process restarts.
- **L3 (CDN)**: Edge caching for static content and API responses. Geographically distributed for lowest user-perceived latency.

Cache invalidation strategies:
- **TTL (Time-To-Live)**: Simple but may serve stale data within TTL window
- **Write-through**: Update cache on every write. Always consistent but increases write latency
- **Cache-aside with invalidation**: Delete cache entry on write; next read populates fresh data
- **Event-driven invalidation**: Database change events trigger cache invalidation

### Load Balancing

Layer 4 (TCP) vs Layer 7 (HTTP) load balancing:
- L4 makes routing decisions based on IP and port; lower latency, no request inspection
- L7 inspects HTTP headers, paths, and cookies; enables content-based routing, session stickiness

Algorithms:
- **Round-robin**: Simple rotation through servers. Even distribution but ignores server load.
- **Least connections**: Routes to server with fewest active connections. Better for varying request durations.
- **Weighted round-robin**: Assigns different weights based on server capacity.
- **Consistent hashing**: Maps requests to servers using a hash ring. Adding/removing servers only redirects 1/N of requests.

Health checks:
- **Active**: Load balancer periodically sends probes (HTTP GET /health)
- **Passive**: Monitor real request success/failure rates
- **Graceful degradation**: Mark unhealthy servers as draining; complete in-flight requests before removal
