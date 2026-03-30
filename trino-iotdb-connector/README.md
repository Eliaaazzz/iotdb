# Trino-IoTDB Connector PoC

A proof-of-concept [Trino](https://trino.io/) connector for [Apache IoTDB](https://iotdb.apache.org/),
built on Trino's `trino-base-jdbc` framework with IoTDB's JDBC driver in
Table Model mode (`sql_dialect=table`).

> **Status:** Pre-coding-period PoC — validates the JDBC-first integration
> path described in the
> [GSoC 2026 proposal](https://github.com/Eliaaazzz/iotdb/tree/trino-connector-poc).

## What the PoC Validates

| Area | What Is Validated | Files |
|------|-------------------|-------|
| Plugin scaffolding | `IoTDBPlugin` loads via `META-INF/services`, `DriverConnectionFactory` opens connections to IoTDB in Table Model mode, Guice wiring resolves all bindings. Compiles and packages against Trino 449. | 6 source classes |
| Metadata & type mapping | `listSchemas()` via `SHOW DATABASES` (filtering `information_schema`). `toColumnMapping()` covers all 10 IoTDB types (BOOLEAN, INT32, INT64, FLOAT, DOUBLE, TEXT, STRING, BLOB, TIMESTAMP, DATE). Dynamic timestamp precision detection via `SHOW VARIABLES` fallback. | 10 type-mapping tests |
| Pushdown | LIMIT pushdown (`isLimitGuaranteed()`) and Top-N pushdown (`isTopNGuaranteed()`) enabled. Predicate pushdown scaffolded through `BaseJdbcClient`'s domain-based mechanism. | verified at compile time |
| Integration-test infra | `TestingIoTDBServer` (Testcontainers + IoTDB Docker), `IoTDBQueryRunner` (`DistributedQueryRunner` factory), `TestIoTDBConnectorTest` extending `BaseJdbcConnectorTest` (400+ inherited tests). | 5 test classes |

**Total:** 14 files, 1 231 lines of code.

## Architecture

```
IoTDBPlugin (extends JdbcPlugin)
  └─ registers connector name "iotdb"
  └─ combines IoTDBClientModule + IoTDBConnectionModule

IoTDBClient (extends BaseJdbcClient)
  ├─ listSchemas()        → SHOW DATABASES
  ├─ toColumnMapping()    → 10 IoTDB types → Trino types
  ├─ getTimestampPrecision() → SHOW VARIABLES → TIMESTAMP(3/6/9)
  ├─ isLimitGuaranteed()  → true
  └─ isTopNGuaranteed()   → true

IoTDBConnectionModule
  └─ @Provides @ForBaseJdbc ConnectionFactory
     └─ injects sql_dialect=table into JDBC properties
```

## Project Structure

```
trino-iotdb-connector/
├── pom.xml
├── etc/catalog/iotdb.properties           # Example catalog config
├── src/main/java/io/trino/plugin/iotdb/
│   ├── IoTDBPlugin.java                   # JdbcPlugin entry point
│   ├── IoTDBClientModule.java             # Guice: binds IoTDBClient
│   ├── IoTDBConnectionModule.java         # Guice: ConnectionFactory + sql_dialect=table
│   ├── IoTDBClient.java                   # BaseJdbcClient: metadata, types, pushdown
│   ├── IoTDBConfig.java                   # Connector config properties
│   └── IoTDBSessionProperties.java        # Session-level toggles
├── src/main/resources/
│   └── META-INF/services/io.trino.spi.Plugin
└── src/test/java/io/trino/plugin/iotdb/
    ├── TestingIoTDBServer.java            # Testcontainers IoTDB wrapper
    ├── IoTDBQueryRunner.java              # DistributedQueryRunner factory
    ├── TestIoTDBPlugin.java               # Smoke test: plugin loads
    ├── TestIoTDBConnectorTest.java        # 400+ inherited BaseJdbcConnectorTest
    └── TestIoTDBTypeMapping.java          # Type round-trip tests (10 IoTDB types)
```

## Prerequisites

- **JDK 22+** (Trino 449 class files require Java 22)
- **Maven 3.9+**
- **Docker** (for integration tests via Testcontainers)
- **IoTDB JDBC driver** installed in local Maven repo

## Build

```bash
# 1. Install IoTDB JDBC driver to local Maven cache
cd /path/to/iotdb
./mvnw install -pl iotdb-client/jdbc -am -DskipTests -Drat.skip=true

# 2. Build the connector
./mvnw -f trino-iotdb-connector/pom.xml package -DskipTests
```

The built JAR is at `target/trino-iotdb-1.0-SNAPSHOT.jar`.

## Run Tests

```bash
# Unit + integration tests (requires Docker running)
./mvnw -f trino-iotdb-connector/pom.xml test

# Plugin smoke test only
./mvnw -f trino-iotdb-connector/pom.xml test -Dtest=TestIoTDBPlugin

# Type mapping round-trips (requires Docker)
./mvnw -f trino-iotdb-connector/pom.xml test -Dtest=TestIoTDBTypeMapping

# Full connector test suite — 400+ inherited tests (requires Docker)
./mvnw -f trino-iotdb-connector/pom.xml test -Dtest=TestIoTDBConnectorTest
```

## Deploy to Trino

```bash
# Copy plugin JAR + dependencies to Trino's plugin directory
mkdir -p <trino-home>/plugin/iotdb/
cp target/trino-iotdb-1.0-SNAPSHOT.jar <trino-home>/plugin/iotdb/
cp target/dependency/*.jar <trino-home>/plugin/iotdb/

# Add catalog config
cp etc/catalog/iotdb.properties <trino-home>/etc/catalog/
# Edit connection-url to point to your IoTDB instance
```

## Example Queries

```sql
-- List IoTDB databases as Trino schemas
SHOW SCHEMAS FROM iotdb;

-- List tables in a database
SHOW TABLES FROM iotdb.factory;

-- Query sensor data
SELECT time, device_id, temperature, humidity
FROM iotdb.factory.sensors
WHERE device_id = 'dev-001'
ORDER BY time DESC
LIMIT 10;

-- Column metadata
DESCRIBE iotdb.factory.sensors;
```

## Catalog Configuration

```properties
# etc/catalog/iotdb.properties
connector.name=iotdb
connection-url=jdbc:iotdb://iotdb-host:6667
connection-user=root
connection-password=root
```

## Key Design Decisions

1. **`sql_dialect=table` injected programmatically** — users cannot accidentally
   omit it; `IoTDBConnectionModule` sets it in JDBC connection properties.

2. **Timestamp precision via `SHOW VARIABLES`** — `IoTDBConnection.unwrap()` is
   unsupported, so `getTimeFactor()` is inaccessible. The connector falls back to
   `SHOW VARIABLES` and caches the cluster-wide `timestamp_precision` setting.

3. **Standalone Maven module** — IoTDB compiles at Java 8, Trino requires Java 22+.
   This module is intentionally NOT a child of `iotdb-parent`.

## License

Apache License 2.0
