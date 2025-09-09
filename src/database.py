import configparser
import sys
import os
import time
from src.logger import Logger
import clickhouse_connect

SHOW_LOG = True


class ClickHouseClient:
    def __init__(self, host: str, port: int, user: str, password: str):
        self.logger = Logger(show=SHOW_LOG).get_logger(__name__)
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.client = None

    def connect(self):
        for i in range(5):
            try:
                self.client = clickhouse_connect.get_client(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password
                )
                self.logger.info(f"Connected to ClickHouse at {self.host}:{self.port}")
                return True
            except Exception as e:
                self.logger.warning(f"Attempt {i+1}: Error connecting to ClickHouse: {e}")
                time.sleep(3)
        self.logger.error("Failed to connect to ClickHouse after 5 attempts")
        sys.exit(1)

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
            self.logger.info("Closed connection to ClickHouse")

    def execute_query(self, query: str):
        if not self.client:
            self.logger.error("No connection to ClickHouse")
            raise RuntimeError("No connection to ClickHouse")
        
        try:
            result = self.client.query(query)
            return result
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            raise
        
    def create_table(self, table_name: str = "predictions"):
        query = f"""CREATE TABLE IF NOT EXISTS {table_name} (
            timestamp DateTime DEFAULT now(),
            message String,
            prediction String
        ) ENGINE = MergeTree()
        ORDER BY timestamp;"""
        
        try:
            self.client.command(query)
            self.logger.info(f"Table `{table_name}` is ready")
        except Exception as e:
            self.logger.error(f"Failed to create table: {e}")
            raise

    def insert_data(self, table_name: str, message: str, prediction: str):
        try:
            self.client.insert(
                table=table_name,
                data=[(message, prediction)],
                column_names=["message", "prediction"]
            )
            self.logger.info(f"Inserted prediction into `{table_name}`")
        except Exception as e:
            self.logger.error(f"Failed to insert data: {e}")
            raise

    def get_data(self, table_name: str, limit: int = 10):
        try:
            query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT {limit}"
            result = self.client.query(query)
            self.logger.info(f"Fetched {len(result.result_rows)} rows from `{table_name}`")
            return result.result_rows
        except Exception as e:
            self.logger.error(f"Failed to fetch data: {e}")
            raise
    