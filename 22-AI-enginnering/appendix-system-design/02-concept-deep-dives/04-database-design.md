---
module: AI Engineering
topic: Concept Deep Dives
subtopic: Database Design
status: unread
tags: [system-design, databases, sharding, replication, indexing, transactions]
---
# Database Design

```mermaid
flowchart TD
    W[Writes] --> P[(Primary)]
    P -->|WAL / binlog| R1[(Replica 1)]
    P -->|async| R2[(Replica 2)]
    Rd[Reads] --> R1 & R2
    subgraph Sharded
      Router{Shard router<br/>hash(user_id)} --> S1[(Shard 1)] & S2[(Shard 2)] & S3[(Shard 3)]
    end
```

## 1. Data Modeling

- Start from **access patterns**, not entities. List the top 3–5 queries and their QPS.
- **Relational:** normalize to 3NF by default; denormalize selectively for hot reads.
- **NoSQL (DynamoDB/Cassandra):** design one table per query (query-first modeling). Partition key = distribution + locality; sort/clustering key = ordering within partition.
- **Document:** embed when data is read together and bounded (order + line items); reference when unbounded (user → posts).

### Example: Chat messages in Cassandra
```sql
CREATE TABLE messages (
  conversation_id uuid,
  bucket int,              -- e.g. month, bounds partition size
  message_id timeuuid,
  sender_id uuid,
  body text,
  PRIMARY KEY ((conversation_id, bucket), message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);
```
Partition = one conversation-month; rows sorted newest-first → "load last 50 messages" is a single-partition read.

## 2. Indexing

| Index | Good for | Notes |
|---|---|---|
| B-tree | Equality, range, ORDER BY | Default |
| Hash | Equality only | |
| Composite `(a, b, c)` | Queries on a; a,b; a,b,c (leftmost prefix) | Order columns by equality first, then range |
| Covering | Query served from index alone | `INCLUDE` columns |
| Partial | `WHERE status='active'` | Smaller |
| GIN / inverted | JSONB, arrays, full-text | |
| GiST / R-tree | Geo (PostGIS) | |
| Vector (HNSW/IVFFlat) | pgvector similarity | |

Cost: every index slows writes and uses storage. Index for the queries you have.

## 3. Transactions and Isolation

**ACID:** Atomicity, Consistency, Isolation, Durability.

| Isolation | Prevents | Allows |
|---|---|---|
| Read uncommitted | — | Dirty reads |
| Read committed (Postgres default) | Dirty reads | Non-repeatable reads, phantoms, lost updates |
| Repeatable read (MySQL default; Postgres = snapshot) | Non-repeatable reads | Write skew (in snapshot isolation) |
| Serializable | Everything | Lower throughput, retries |

**Concurrency control:**
- **Pessimistic:** `SELECT … FOR UPDATE` — lock rows. Good under high contention.
- **Optimistic:** `version` column; `UPDATE … SET v=v+1 WHERE id=? AND v=?`; retry if 0 rows. Good under low contention.
- **Atomic ops:** `UPDATE inventory SET qty = qty - 1 WHERE id=? AND qty > 0` — simplest correct oversell prevention.

## 4. Replication

| Mode | How | Pros | Cons |
|---|---|---|---|
| **Single-leader** | All writes to primary, replicate to followers | Simple, no conflicts | Write bottleneck; failover complexity |
| **Multi-leader** | Several primaries (per region) | Local writes in each region | Conflicts (LWW, CRDTs, app merge) |
| **Leaderless** (Dynamo) | Write to W of N, read from R of N; `R + W > N` → overlap | High availability | Sloppy quorums, read repair, hinted handoff |

- **Sync vs async:** sync = no data loss, higher write latency; async = lag (seconds) and possible loss on failover. Common: semi-sync (1 sync replica).
- **Replication lag problems:** read-your-writes (route user to primary for N sec after write, or use LSN tokens), monotonic reads (sticky replica).

## 5. Partitioning / Sharding

| Strategy | How | Pros | Cons |
|---|---|---|---|
| **Range** | key ranges per shard | Range scans | Hot spots (time-based keys) |
| **Hash** | hash(key) mod N or consistent hashing | Even distribution | No range scans; resharding with mod N moves everything |
| **Directory / lookup** | Mapping table key → shard | Flexible, move tenants | Lookup service = SPOF/latency |
| **Geo** | By region | Locality, data residency | Uneven |

**Choosing a shard key:** high cardinality, even distribution, and colocates data accessed together (e.g., `user_id` for user-centric apps, `tenant_id` for SaaS). Avoid monotonic keys (timestamps) as range keys.

**Problems:**
- **Cross-shard queries** → scatter-gather; avoid via denormalization or secondary index service.
- **Cross-shard transactions** → 2PC or sagas ([Distributed systems](../02-concept-deep-dives/05-distributed-systems.md)).
- **Hot shards** → split shard, add salt, or isolate whale tenants.
- **Resharding** → consistent hashing / many logical shards mapped to fewer physical (e.g., 4096 logical → 16 physical), move logical shards online with dual-writes + backfill.

## 6. Denormalization & Derived Data

- Counters stored on the parent (`post.like_count`) updated async.
- Materialized views / read models (CQRS): write model in Postgres, read models in Redis/Elasticsearch fed by CDC.
- Keep **one source of truth**; everything else is derived and rebuildable.

## 7. Schema Evolution & Migrations

- Expand → migrate → contract: add nullable column, dual-write, backfill, switch reads, drop old.
- Online schema change tools: gh-ost, pt-online-schema-change.
- Never do a blocking `ALTER` on a large hot table.

## 8. Storage Engines

B-tree (read-optimized) vs LSM (write-optimized) — see [Data Structures](../01-foundations/05-data-structures.md#7-b-tree-vs-lsm-tree). Row vs columnar: OLTP rows; OLAP columns (compression, vectorized scans).

## 9. Time-Series and Hot/Cold Tiering

- Partition by time (Postgres declarative partitioning, TimescaleDB hypertables).
- Recent data on SSD/hot DB; older to S3 (Parquet) queried by Athena/Trino.
- TTL-based deletion via dropping partitions (cheap) not `DELETE` (expensive).

## 10. Database Choices for AI Systems

| Data | Store |
|---|---|
| Conversations, messages | Postgres (moderate scale) or Cassandra/DynamoDB (huge scale), partitioned by conversation |
| Documents for RAG | S3 (raw) + Postgres (metadata, ACLs) + vector index (chunks) + BM25 index |
| Embeddings | pgvector / dedicated vector DB; always store `embedding_model_version` |
| Prompt templates & versions | Postgres / Git-backed registry |
| Traces, eval results | ClickHouse / OLAP, or observability vendor |
| Usage/billing tokens | Kafka → aggregated in OLAP, reconciled to Postgres ledger |
| Agent state / checkpoints | Postgres (LangGraph checkpointer) / Temporal |

## 11. Interview Checklist
- [ ] Listed top queries and picked DB to match
- [ ] Primary key / partition key justified
- [ ] Indexes for each query
- [ ] Replication mode and lag handling
- [ ] Sharding strategy *if numbers require it*
- [ ] Concurrency control for contended writes
- [ ] Retention / tiering for big data
