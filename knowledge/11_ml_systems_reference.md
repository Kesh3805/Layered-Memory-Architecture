# Machine Learning Systems: A Comprehensive Technical Reference

## Neural Network Architectures and Training

### Feedforward Neural Networks

A feedforward neural network (FNN) consists of layers of neurons where information moves in one direction from input to output. Each neuron computes a weighted sum of its inputs, applies a bias, and passes the result through an activation function. The universal approximation theorem states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of Rn, given appropriate activation functions.

The architecture is defined by the number of layers, number of neurons per layer, choice of activation function, and weight initialization strategy. Common activation functions include ReLU (Rectified Linear Unit), which computes max(0, x); sigmoid, which maps values to (0, 1); and tanh, which maps to (-1, 1). The dying ReLU problem occurs when neurons get stuck outputting zero for any input, effectively "dying" during training. Leaky ReLU addresses this by allowing a small negative slope.

Weight initialization is critical for training stability. Xavier/Glorot initialization sets weights from a distribution scaled by 1/sqrt(n_in), maintaining variance across layers for sigmoid/tanh activations. He initialization uses 1/sqrt(2/n_in) and is designed for ReLU activations. Poor initialization can lead to vanishing or exploding gradients.

### Convolutional Neural Networks

Convolutional Neural Networks (CNNs) exploit the spatial structure of data through three key operations: convolution, pooling, and nonlinearity. The convolution operation slides a learned filter (kernel) across the input, computing dot products to produce feature maps. Key hyperparameters include kernel size (typically 3x3 or 5x5), stride (step size), padding (to control output dimensions), and number of filters (depth of output).

Pooling layers reduce spatial dimensions while retaining important features. Max pooling takes the maximum value in each pooling window, while average pooling computes the mean. Global average pooling collapses each feature map to a single value, reducing parameters.

Modern architectures include ResNet (residual connections that enable training of very deep networks by learning residual functions), DenseNet (dense connections where each layer receives inputs from all preceding layers), and EfficientNet (compound scaling of depth, width, and resolution). Skip connections in ResNet solve the degradation problem where deeper networks paradoxically perform worse than shallower ones.

### Recurrent Neural Networks and Transformers

Recurrent Neural Networks (RNNs) process sequential data by maintaining a hidden state that is updated at each time step. The vanilla RNN update rule is h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h). The key limitation is the vanishing gradient problem: gradients diminish exponentially through time steps, making it difficult to learn long-range dependencies.

Long Short-Term Memory (LSTM) networks address this with a gating mechanism consisting of forget gate, input gate, and output gate. The cell state acts as a conveyor belt, allowing information to flow unchanged across time steps. The forget gate decides what to discard from cell state, the input gate decides what new information to store, and the output gate determines what to output.

The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), replaces recurrence entirely with self-attention. Multi-head attention computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V, where Q, K, V are queries, keys, and values projected from input embeddings. Multi-head attention runs h parallel attention functions, concatenates results, and projects back.

Positional encoding adds position information since self-attention is permutation-invariant. Sinusoidal encoding uses sin and cos functions of different frequencies, while learned positional embeddings are trained alongside model weights. Rotary Position Embedding (RoPE) encodes position through rotation matrices, enabling better length generalization.

## Optimization and Regularization

### Gradient Descent Variants

Stochastic Gradient Descent (SGD) updates parameters using gradients computed on mini-batches rather than the full dataset. The update rule is θ_{t+1} = θ_t - η * ∇L(θ_t), where η is the learning rate. Momentum extends SGD by accumulating an exponentially decaying moving average of past gradients: v_t = β * v_{t-1} + ∇L(θ_t), θ_{t+1} = θ_t - η * v_t.

Adam (Adaptive Moment Estimation) combines momentum with per-parameter adaptive learning rates. It maintains first moment (mean) and second moment (uncentered variance) estimates: m_t = β1 * m_{t-1} + (1-β1) * g_t, v_t = β2 * v_{t-1} + (1-β2) * g_t^2. Bias correction compensates for initialization: m_hat = m_t / (1-β1^t), v_hat = v_t / (1-β2^t). The update is θ_{t+1} = θ_t - η * m_hat / (sqrt(v_hat) + ε).

AdamW decouples weight decay from the gradient update, applying it directly to parameters rather than through the gradient. This distinction matters for adaptive methods because L2 regularization and weight decay are equivalent for SGD but not for Adam. Learning rate scheduling strategies include cosine annealing, warm restarts, and linear warmup followed by decay.

### Regularization Techniques

Dropout randomly sets a fraction p of activations to zero during training, acting as an ensemble of exponentially many sub-networks. At inference time, activations are scaled by (1-p) to maintain expected values. DropConnect extends this idea by dropping connections (weights) rather than activations.

Batch Normalization normalizes activations within each mini-batch to zero mean and unit variance, then applies learned scale and shift parameters. It reduces internal covariate shift and allows higher learning rates. Layer Normalization normalizes across features rather than batch dimension, making it suitable for variable-length sequences and small batch sizes. Group Normalization divides channels into groups and normalizes within each group.

Label smoothing replaces hard targets (one-hot) with soft targets by redistributing a small probability mass ε across non-target classes. Instead of [1, 0, 0], the target becomes [1-ε, ε/2, ε/2]. This prevents the model from becoming overconfident and improves calibration.

## Information Retrieval Systems

### Vector Search and Similarity

Vector search finds nearest neighbors in high-dimensional embedding spaces. The fundamental operations are:
- Cosine similarity: cos(a, b) = (a · b) / (||a|| * ||b||), measuring angle between vectors
- Euclidean distance: L2 norm of the difference vector
- Inner product: dot product, equivalent to cosine when vectors are normalized

Approximate Nearest Neighbor (ANN) algorithms trade exact accuracy for speed. HNSW (Hierarchical Navigable Small World) builds a multi-layer graph where each layer contains a subset of nodes. Search proceeds from the top layer (fewest nodes) to the bottom (all nodes), greedily traversing edges to find nearest neighbors. Key parameters are M (number of bi-directional links per node, typically 16-64), ef_construction (search width during index building), and ef_search (search width at query time).

IVF (Inverted File Index) partitions the vector space into Voronoi cells using k-means clustering. At query time, only the nprobe nearest cells are searched. IVF+PQ combines this with Product Quantization, which compresses vectors by splitting them into sub-vectors and quantizing each sub-vector independently. IVFADC (Asymmetric Distance Computation) computes exact distances between the query and compressed database vectors.

### BM25 and Lexical Search

BM25 (Best Matching 25) is a probabilistic ranking function based on term frequency, inverse document frequency, and document length normalization. The formula is:

score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

Where f(qi, D) is the term frequency of qi in document D, |D| is the document length, avgdl is the average document length, k1 controls term frequency saturation (typically 1.2-2.0), and b controls length normalization (typically 0.75).

IDF is computed as log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1), where N is the total number of documents and n(qi) is the number of documents containing qi. This formula avoids negative IDF values for very common terms.

BM25 excels at exact term matching and is particularly effective for queries containing rare technical terms, product names, or specific identifiers. However, it fails to capture semantic similarity — "car" and "automobile" are treated as completely different terms.

### Hybrid Search with Reciprocal Rank Fusion

Hybrid search combines lexical (BM25) and semantic (vector) retrieval to leverage the strengths of both approaches. Reciprocal Rank Fusion (RRF) merges ranked lists from multiple retrieval systems using the formula:

RRF_score(d) = Σ 1 / (k + rank_i(d))

Where k is a constant (typically 60) that dampens the contribution of low-ranked documents, and rank_i(d) is the rank of document d in the i-th ranking. RRF has several desirable properties: it requires no training, is robust to outliers, handles missing documents gracefully, and works across systems with incomparable score distributions.

The key advantage of hybrid search emerges when corpora exceed approximately 10,000 documents. At this scale, the vocabulary becomes diverse enough that BM25 captures exact-match signals that vector search may miss, while vector search handles paraphrased and semantically related queries that BM25 cannot match. On small corpora (<1000 documents), the top-k cosine matches are nearly optimal, and adding BM25 introduces noise without benefit.

### Cross-Encoder Reranking

Cross-encoder rerankers take a query-document pair as input and produce a relevance score through a full Transformer forward pass. Unlike bi-encoders (which embed query and document independently), cross-encoders allow full token-level interaction between query and document tokens, enabling much more accurate relevance estimation.

The architecture typically uses a BERT-like model fine-tuned on relevance datasets (MS MARCO, Natural Questions). The input is formatted as [CLS] query [SEP] document [SEP], and the [CLS] token representation is projected to a scalar relevance score. Models like cross-encoder/ms-marco-MiniLM-L-6-v2 achieve strong performance with relatively low latency (~5ms per pair on GPU).

The two-stage retrieve-then-rerank pipeline first uses a fast retrieval method (BM25, vector search, or hybrid) to retrieve top-N candidates (N=20-100), then reranks these candidates with the cross-encoder. This reduces the computational cost from O(|corpus|) cross-encoder calls to O(N), making it practical for production systems.

Reranking typically improves MRR (Mean Reciprocal Rank) by 15-30% on standard benchmarks, with the largest gains on queries where the initial retrieval returns relevant documents at lower ranks. The main tradeoff is latency: each cross-encoder call takes 5-30ms depending on sequence length and hardware.

## Evaluation Metrics for Information Retrieval

### Precision, Recall, and F1

Precision measures the fraction of retrieved documents that are relevant: P = |relevant ∩ retrieved| / |retrieved|. Recall measures the fraction of relevant documents that are retrieved: R = |relevant ∩ retrieved| / |relevant|. F1 is the harmonic mean: F1 = 2PR / (P+R).

Precision@K (P@K) considers only the top-K results. For K=5, if 3 of the top 5 are relevant, P@5 = 0.6. This is the most intuitive metric for end-user experience.

### Mean Reciprocal Rank (MRR)

MRR measures how quickly users find their first relevant result:

MRR = (1/|Q|) * Σ 1/rank_i

Where rank_i is the position of the first relevant document for query i. If the first relevant document is at position 1, the reciprocal rank is 1.0. At position 2, it's 0.5. At position 5, it's 0.2. If no relevant document is found, the reciprocal rank is 0.

MRR ranges from 0 to 1, with higher values indicating that relevant documents appear earlier in the ranking. An MRR of 0.78 means that, on average, the first relevant document appears between positions 1 and 2.

### Normalized Discounted Cumulative Gain (NDCG)

NDCG accounts for graded relevance (not just binary) and position bias:

DCG@K = Σ_{i=1}^{K} (2^{rel_i} - 1) / log2(i + 1)

NDCG@K = DCG@K / IDCG@K, where IDCG is the DCG of the ideal ranking.

### Mean Average Precision (MAP)

Average Precision for a single query is the mean of precision values computed at each position where a relevant document appears. MAP averages AP across all queries. MAP rewards systems that rank relevant documents higher and is a single-number summary of the precision-recall curve.

## Natural Language Processing

### Tokenization and Subword Models

Byte-Pair Encoding (BPE) starts with individual characters and iteratively merges the most frequent pair of adjacent tokens. The vocabulary size is a hyperparameter controlling the tradeoff between vocabulary coverage and sequence length. GPT-2 uses BPE with a vocabulary of 50,257 tokens.

WordPiece (used by BERT) is similar to BPE but selects merges that maximize the likelihood of the training data rather than frequency. SentencePiece operates on raw text without pre-tokenization, treating the input as a sequence of Unicode characters.

Unigram Language Model tokenization starts with a large vocabulary and iteratively removes tokens that least impact the language model likelihood. This top-down approach contrasts with BPE's bottom-up construction.

### Embedding Models for Retrieval

Sentence embeddings map variable-length text to fixed-dimensional vectors. Key models include:

- **BAAI/bge-base-en-v1.5** (768-dim): State-of-the-art on MTEB benchmark. Uses asymmetric training with query instructions for optimal retrieval performance.
- **all-mpnet-base-v2** (768-dim): General-purpose symmetric model from sentence-transformers. Good balance of quality and speed.
- **all-MiniLM-L6-v2** (384-dim): Lightweight model, 5x faster inference with moderate quality loss. Suitable for high-throughput applications.
- **E5-large-v2** (1024-dim): Microsoft's embedding model, strong on asymmetric retrieval tasks.
- **nomic-embed-text-v1.5** (768-dim): Open-source model with Matryoshka representation learning, allowing flexible dimensionality reduction.

Asymmetric retrieval models are trained with different prefixes for queries and documents. For bge models, queries are prefixed with "Represent this sentence for searching relevant passages:" while documents use no prefix. This asymmetry improves retrieval performance by specializing the embedding space for the retrieval task.

### Chunking Strategies

Document chunking divides long texts into segments suitable for embedding and retrieval. Strategies include:

1. **Fixed-size chunking**: Split at character or token boundaries with configurable overlap. Simple but may split mid-sentence or mid-concept.

2. **Recursive character splitting**: Split on paragraph boundaries first, then sentences, then characters. Preserves natural document structure.

3. **Semantic chunking**: Embed individual sentences, compute pairwise cosine similarities between consecutive sentences, and split at points where similarity drops below a threshold (or at the Nth percentile of distance distribution). This produces chunks that are semantically coherent.

4. **Agentic chunking**: Use an LLM to decide chunk boundaries based on topic coherence. Most expensive but highest quality.

Chunk size affects retrieval precision and recall. Smaller chunks (200-300 tokens) increase precision by reducing noise but may lose context. Larger chunks (500-1000 tokens) preserve context but increase noise. Overlap between chunks (typically 10-20% of chunk size) ensures that information at chunk boundaries is not lost.

## Database Systems and Query Optimization

### PostgreSQL Internals

PostgreSQL uses a multi-version concurrency control (MVCC) system where each transaction sees a snapshot of the database at the time it started. This eliminates read locks and enables high concurrency. Each row version (tuple) carries xmin and xmax transaction IDs indicating when it was created and deleted.

The query planner uses cost-based optimization to choose execution plans. It estimates the cost of different access paths (sequential scan, index scan, bitmap scan, index-only scan) based on statistics collected by ANALYZE. The planner considers CPU cost, I/O cost, and network cost for distributed queries.

Index types in PostgreSQL include B-tree (default, for equality and range queries), Hash (equality only, faster than B-tree for point lookups), GiST (Generalized Search Tree, for spatial and full-text data), GIN (Generalized Inverted Index, for array and full-text containment), and BRIN (Block Range Index, for naturally ordered data like timestamps).

### pgvector for Vector Similarity Search

pgvector extends PostgreSQL with vector data types and similarity search operators. It supports three distance functions: L2 distance (<->), inner product (<#>), and cosine distance (<=>). Vectors are stored as binary data alongside regular columns.

Index options include:
- **IVFFlat**: Partitions vectors into lists using k-means. Set lists = sqrt(n_rows) for balanced performance. Requires periodic reindexing as data changes. Probes parameter controls accuracy/speed tradeoff.
- **HNSW**: Builds a proximity graph for efficient approximate nearest neighbor search. Set m = 16-64 (connections per node) and ef_construction = 64-200. More expensive to build but better query performance and no reindexing needed.

Performance tuning:
- `maintenance_work_mem`: Set high during index creation (e.g., 2GB for 1M vectors)
- `work_mem`: Controls memory for sort/hash operations in queries
- `effective_cache_size`: Tells the planner how much memory is available for caching
- `shared_buffers`: PostgreSQL's internal buffer pool, typically 25% of system RAM
- `max_parallel_workers_per_gather`: Enable parallel query execution for large scans

### Connection Pooling and Connection Management

Database connections are expensive to establish (TCP handshake, authentication, process creation). Connection pooling maintains a pool of open connections that can be reused by application threads.

PgBouncer is a lightweight connection pooler for PostgreSQL supporting three modes: session pooling (connection held for the entire session), transaction pooling (connection returned after each transaction), and statement pooling (connection returned after each statement). Transaction pooling is most common for web applications.

In Python, psycopg2's ThreadedConnectionPool manages a pool of database connections for multi-threaded applications. The pool is initialized with minconn and maxconn parameters. Connections are obtained with pool.getconn() and returned with pool.putconn(). For async applications, asyncpg provides native async connection pooling.

## Distributed Systems

### Consensus and Replication

The Raft consensus algorithm elects a leader that manages log replication to followers. Safety is guaranteed: at most one leader per term, committed entries are durable, and the state machine applies entries in log order. The leader sends AppendEntries RPCs to replicate log entries and heartbeats to maintain authority.

PostgreSQL supports streaming replication where WAL (Write-Ahead Log) records are sent from primary to standby servers in real-time. Synchronous replication guarantees that transactions are committed on at least one standby before acknowledging to the client, at the cost of increased latency. Asynchronous replication has lower latency but risks data loss on primary failure.

### Caching Strategies

Cache-aside (lazy loading): Application checks cache first; on miss, loads from database and populates cache. Advantages: only requested data is cached, cache misses don't break the application. Disadvantages: initial requests are always slow, data can become stale.

Write-through: Every write goes to both cache and database. Ensures cache is always consistent but adds write latency. Write-behind (write-back) batches writes to the database, reducing database load at the risk of data loss on cache failure.

Redis supports multiple eviction policies: volatile-lru (evict least recently used keys with TTL), allkeys-lru (evict any LRU key), volatile-ttl (evict keys with shortest TTL), and noeviction (return error on memory limit). For RAG applications, embedding cache with TTL significantly reduces repeated embedding computation.

## Software Engineering Practices

### Testing Strategies

Unit tests verify individual functions in isolation. Mock objects replace real dependencies (databases, APIs) with controlled substitutes. Property-based testing (Hypothesis in Python) generates random inputs to find edge cases that manual test cases miss.

Integration tests verify that components work together. For database-backed applications, use test databases with known state, transaction rollback for test isolation, and fixtures for common data patterns. Docker-based test databases ensure consistent environments.

Load testing measures system performance under stress. Key metrics include throughput (requests per second), latency percentiles (p50, p95, p99), and error rate. Tools include Locust (Python), k6 (JavaScript), and Apache JMeter. Chaos engineering (Netflix's Chaos Monkey) randomly introduces failures to test system resilience.

### Observability and Monitoring

The three pillars of observability are metrics, logs, and traces. Metrics are numerical measurements aggregated over time (counters, gauges, histograms). Logs are timestamped text records of discrete events. Traces follow a request through distributed system components.

Structured logging formats log entries as JSON with consistent fields (timestamp, level, service, trace_id, message). This enables efficient log parsing, filtering, and aggregation. Log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) control verbosity in different environments.

OpenTelemetry provides a vendor-neutral standard for telemetry data collection. Auto-instrumentation libraries for Python, Java, and Node.js automatically capture traces for common frameworks. Custom spans and attributes provide application-specific context.
