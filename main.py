#Download RAW data
from src.download import download_puzzles
from src.download import download_games
#Preprocess RAW data
from src.preprocess import preprocess_puzzles
from src.preprocess import parse_games
#Validate and clean preprocessed data
from src.preprocess import clean_games
from src.preprocess import clean_puzzles

def main():

    download_puzzles.main()
    download_games.main()
    preprocess_puzzles.main()
    parse_games.main()
    clean_games.main()
    clean_puzzles.main()


if __name__ == "__main__":
    main()