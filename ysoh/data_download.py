import kagglehub 

def main():
    path = kagglehub.competition_download('rs-18-track-a')
    print("Path to competition files:", path)

    path = kagglehub.competition_download('rs-18-track-b')
    print("Path to competition files:", path)

if __name__ == "__main__":
    main()