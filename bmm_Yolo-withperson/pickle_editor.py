import argparse
import pickle

def read_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: Unable to read pickle file '{file_path}'.")
        print(f"Details: {str(e)}")
        return None

def print_pickle_contents(file_path):
    data = read_pickle_file(file_path)
    if data is not None:
        print("Pickle file contents:")
        print(data)

def edit_pickle(file_path):
    while True:
        data = read_pickle_file(file_path)
        if data is not None:
            print("Pickle file contents:")
            print(data)

        user_input = input("Do you want to edit or add parameters? (yes/no): ").lower()

        if user_input == 'no':
            break
        elif user_input == 'yes':
            key = input("Enter the parameter you want to edit or add: ")
            value = input("Enter the new value: ")

            if key in data:
                print(f"Editing existing parameter '{key}'.")
            else:
                print(f"Adding new parameter '{key}'.")

            data[key] = value

            # Save the updated data back to the pickle file
            with open(file_path, 'wb') as file:
                pickle.dump(data, file)
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def main():
    parser = argparse.ArgumentParser(description="Read and interactively edit a pickle file.")
    parser.add_argument('file_path', help='Path to the pickle file')
    args = parser.parse_args()

    file_path = args.file_path
    print_pickle_contents(file_path)
    edit_pickle(file_path)

if __name__ == "__main__":
    main()
