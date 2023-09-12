import client.domsed_client as dc

if __name__ == "__main__":
    mutation_name = sys.argv[1]
    dc.delete(mutation_name)