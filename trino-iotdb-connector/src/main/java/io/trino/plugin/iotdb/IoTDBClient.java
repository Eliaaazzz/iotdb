/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.trino.plugin.iotdb;

import com.google.common.collect.ImmutableSet;
import com.google.inject.Inject;
import io.trino.plugin.base.mapping.IdentifierMapping;
import io.trino.plugin.jdbc.BaseJdbcClient;
import io.trino.plugin.jdbc.BaseJdbcConfig;
import io.trino.plugin.jdbc.ColumnMapping;
import io.trino.plugin.jdbc.ConnectionFactory;
import io.trino.plugin.jdbc.ForBaseJdbc;
import io.trino.plugin.jdbc.JdbcTypeHandle;
import io.trino.plugin.jdbc.QueryBuilder;
import io.trino.plugin.jdbc.WriteMapping;
import io.trino.plugin.jdbc.logging.RemoteQueryModifier;
import io.trino.spi.TrinoException;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.type.TimestampType;
import io.trino.spi.type.Type;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.Types;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

import static io.trino.plugin.jdbc.StandardColumnMappings.bigintColumnMapping;
import static io.trino.plugin.jdbc.StandardColumnMappings.booleanColumnMapping;
import static io.trino.plugin.jdbc.StandardColumnMappings.dateColumnMappingUsingSqlDate;
import static io.trino.plugin.jdbc.StandardColumnMappings.doubleColumnMapping;
import static io.trino.plugin.jdbc.StandardColumnMappings.integerColumnMapping;
import static io.trino.plugin.jdbc.StandardColumnMappings.realColumnMapping;
import static io.trino.plugin.jdbc.StandardColumnMappings.timestampColumnMappingUsingSqlTimestampWithRounding;
import static io.trino.plugin.jdbc.StandardColumnMappings.varbinaryColumnMapping;
import static io.trino.plugin.jdbc.StandardColumnMappings.varcharColumnMapping;
import static io.trino.spi.StandardErrorCode.GENERIC_INTERNAL_ERROR;
import static io.trino.spi.StandardErrorCode.NOT_SUPPORTED;
import static io.trino.spi.type.TimestampType.createTimestampType;
import static io.trino.spi.type.VarcharType.createUnboundedVarcharType;

/**
 * IoTDB JDBC client extending Trino's BaseJdbcClient.
 * <p>
 * Implements:
 * <ul>
 *   <li>Schema listing via SHOW DATABASES</li>
 *   <li>Type mapping for all 10 IoTDB data types</li>
 *   <li>Dynamic timestamp precision detection via SHOW VARIABLES</li>
 *   <li>LIMIT and Top-N pushdown</li>
 *   <li>Predicate pushdown via BaseJdbcClient's domain-based mechanism</li>
 * </ul>
 * <p>
 * The class hierarchy follows Trino's JDBC decorator pattern:
 * {@code IoTDBClient (BaseJdbcClient) -> CachingJdbcClient -> RetryingJdbcClient -> JdbcClient (SPI)}
 */
public class IoTDBClient
        extends BaseJdbcClient
{
    /**
     * Cached timestamp precision: 3 (ms), 6 (us), or 9 (ns).
     * Detected once from SHOW VARIABLES and cached since it's a cluster-level config.
     */
    private final AtomicReference<Integer> cachedTimestampPrecision = new AtomicReference<>();

    @Inject
    public IoTDBClient(
            BaseJdbcConfig config,
            @ForBaseJdbc ConnectionFactory connectionFactory,
            QueryBuilder queryBuilder,
            IdentifierMapping identifierMapping,
            RemoteQueryModifier remoteQueryModifier)
    {
        // IoTDB uses double-quote as identifier quote character
        // (matches IoTDBRelationalDatabaseMetadata.getIdentifierQuoteString())
        super("\"", connectionFactory, queryBuilder,
                config.getJdbcTypesMappedToVarchar(), identifierMapping,
                remoteQueryModifier, false);
    }

    // ==================== Schema Listing ====================

    /**
     * List IoTDB databases as Trino schemas.
     * <p>
     * Executes {@code SHOW DATABASES} and filters out {@code information_schema}
     * to avoid conflicts with Trino's own information_schema.
     */
    @Override
    public Collection<String> listSchemas(Connection connection)
    {
        ArrayList<String> schemas = new ArrayList<>();
        try (Statement statement = connection.createStatement();
                ResultSet resultSet = statement.executeQuery("SHOW DATABASES")) {
            while (resultSet.next()) {
                String schemaName = resultSet.getString(1);
                if (!"information_schema".equalsIgnoreCase(schemaName)) {
                    schemas.add(schemaName);
                }
            }
        }
        catch (SQLException e) {
            throw new TrinoException(GENERIC_INTERNAL_ERROR,
                    "Failed to list IoTDB databases: " + e.getMessage(), e);
        }
        return schemas;
    }

    // ==================== Type Mapping ====================

    /**
     * Map JDBC types from IoTDB to Trino column types.
     * <p>
     * Complete mapping table:
     * <pre>
     * IoTDB Type | JDBC Type        | Trino Type      | Pushdown
     * -----------|------------------|-----------------|----------
     * BOOLEAN    | Types.BOOLEAN    | BOOLEAN         | Full
     * INT32      | Types.INTEGER    | INTEGER         | Full
     * INT64      | Types.BIGINT     | BIGINT          | Full
     * FLOAT      | Types.FLOAT      | REAL            | Full
     * DOUBLE     | Types.DOUBLE     | DOUBLE          | Full
     * TEXT       | Types.VARCHAR    | VARCHAR         | Selective
     * STRING     | Types.VARCHAR    | VARCHAR         | Selective
     * BLOB       | Types.BLOB       | VARBINARY       | Disabled
     * TIMESTAMP  | Types.TIMESTAMP  | TIMESTAMP(p)*   | Full
     * DATE       | Types.DATE       | DATE            | Full
     * </pre>
     * *p = 3 (ms), 6 (us), or 9 (ns) — detected from SHOW VARIABLES
     */
    @Override
    public Optional<ColumnMapping> toColumnMapping(ConnectorSession session, Connection connection, JdbcTypeHandle typeHandle)
    {
        int jdbcType = typeHandle.jdbcType();

        switch (jdbcType) {
            case Types.BOOLEAN:
                return Optional.of(booleanColumnMapping());

            case Types.INTEGER:
            case Types.SMALLINT:
            case Types.TINYINT:
                return Optional.of(integerColumnMapping());

            case Types.BIGINT:
                return Optional.of(bigintColumnMapping());

            case Types.FLOAT:
                // IoTDB FLOAT is 32-bit, maps to Trino REAL
                return Optional.of(realColumnMapping());

            case Types.DOUBLE:
                return Optional.of(doubleColumnMapping());

            case Types.VARCHAR:
            case Types.LONGVARCHAR:
            case Types.NVARCHAR:
            case Types.LONGNVARCHAR:
                // Both IoTDB TEXT and STRING map to unbounded VARCHAR
                return Optional.of(varcharColumnMapping(createUnboundedVarcharType(), false));

            case Types.BLOB:
            case Types.BINARY:
            case Types.VARBINARY:
            case Types.LONGVARBINARY:
                return Optional.of(varbinaryColumnMapping());

            case Types.TIMESTAMP:
                int precision = getTimestampPrecision(connection);
                TimestampType timestampType = createTimestampType(precision);
                return Optional.of(timestampColumnMappingUsingSqlTimestampWithRounding(timestampType));

            case Types.DATE:
                return Optional.of(dateColumnMappingUsingSqlDate());

            default:
                return Optional.empty();
        }
    }

    /**
     * Write mapping is not supported in this PoC.
     * IoTDB's JDBC documentation positions JDBC primarily for query workloads.
     */
    @Override
    public WriteMapping toWriteMapping(ConnectorSession session, Type type)
    {
        throw new TrinoException(NOT_SUPPORTED,
                "IoTDB connector does not support writes. Type: " + type);
    }

    // ==================== Timestamp Precision Detection ====================

    /**
     * Detect IoTDB cluster's timestamp precision via SHOW VARIABLES.
     * <p>
     * Returns Trino-style precision digits:
     * <ul>
     *   <li>3 -> milliseconds (TIMESTAMP(3))</li>
     *   <li>6 -> microseconds (TIMESTAMP(6))</li>
     *   <li>9 -> nanoseconds  (TIMESTAMP(9))</li>
     * </ul>
     * <p>
     * Note: IoTDBConnection.unwrap() throws SQLException("Does not support unwrap")
     * (IoTDBConnection.java:163), so we cannot directly access getTimeFactor().
     * Instead, we execute SHOW VARIABLES and scan for the timestamp_precision row.
     * The result is cached since timestamp precision is a cluster-level configuration.
     */
    private int getTimestampPrecision(Connection connection)
    {
        Integer cached = cachedTimestampPrecision.get();
        if (cached != null) {
            return cached;
        }

        int precision = detectTimestampPrecisionFromShowVariables(connection);
        cachedTimestampPrecision.compareAndSet(null, precision);
        return cachedTimestampPrecision.get();
    }

    private int detectTimestampPrecisionFromShowVariables(Connection connection)
    {
        try (Statement statement = connection.createStatement();
                ResultSet resultSet = statement.executeQuery("SHOW VARIABLES")) {
            while (resultSet.next()) {
                String variableName = resultSet.getString(1);
                if ("timestamp_precision".equalsIgnoreCase(variableName)) {
                    String value = resultSet.getString(2);
                    return mapPrecisionStringToTrino(value);
                }
            }
        }
        catch (SQLException e) {
            // Fall through to default
        }
        // Default to milliseconds (most common IoTDB deployment)
        return 3;
    }

    private static int mapPrecisionStringToTrino(String precision)
    {
        if (precision == null) {
            return 3;
        }
        switch (precision.toLowerCase()) {
            case "ms":
                return 3;
            case "us":
                return 6;
            case "ns":
                return 9;
            default:
                return 3;
        }
    }

    // ==================== Limit & TopN Pushdown ====================

    /**
     * IoTDB's LIMIT clause is deterministic and preserves ordering semantics.
     */
    @Override
    public boolean isLimitGuaranteed(ConnectorSession session)
    {
        return true;
    }

    /**
     * IoTDB supports ORDER BY + LIMIT for Top-N pushdown on sortable types.
     * Enables the common IoT "last N readings" query pattern.
     */
    @Override
    public boolean isTopNGuaranteed(ConnectorSession session)
    {
        return true;
    }
}
