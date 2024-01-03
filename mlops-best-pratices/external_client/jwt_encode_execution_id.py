"""
A utility script to generate a JWT containing an execution id
"""

import sys

import jwt

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <execution id>")
    exit(1)

execution_id = sys.argv[1]
execution_id_jwt = jwt.encode({"execution_id": execution_id}, "secret", algorithm="HS256")

print(execution_id_jwt)
