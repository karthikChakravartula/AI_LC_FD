from supabase_client import fetch_records
from neo4j import GraphDatabase

def buildGraph() :
    
    contacts = fetch_records("contacts","id,name,email,account_id")
    accounts = fetch_records("accounts","id,name,description,status,notes")

    URI = "neo4j://localhost:7687"
    AUTH = ("neo4j","password")

    with GraphDatabase.driver(URI,auth=AUTH) as driver:

        account_query = """
        UNWIND $batch AS row
        MERGE (a:Account {id: row.id})
        ON CREATE SET 
            a.name = row.name, 
            a.status = row.status, 
            a.description = row.description,
            a.notes = row.notes

            ON MATCH SET 
                a.name = row.name, 
                a.status = row.status, 
                a.description = row.description,
                a.notes = CASE WHEN row.notes IS NOT NULL THEN row.notes ELSE a.notes END
            """

        contact_query = """
                UNWIND $batch AS row
                MERGE (c:Contact {id: row.id})
                SET c.name = row.name,
                    c.email = row.email 

                with c, row
                MATCH(a:Account {id:row.account_id})
                MERGE (c)-[:WORKS_AT]->(a)      
                """
        acc_records = driver.execute_query(account_query, batch=accounts, database_="neo4j").summary
        print(f"Account nodes created : {acc_records.counters.nodes_created}")
        con_records =driver.execute_query(contact_query, batch=contacts, database_="neo4j").summary
        print(f"Contact nodes created : {con_records.counters.nodes_created}")

buildGraph()