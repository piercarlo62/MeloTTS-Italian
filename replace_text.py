import os

def replace_text_in_file(file_path, old_text, new_text):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    # Replace the text
    new_content = file_content.replace(old_text, new_text)

    # Write the changes back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)

if __name__ == "__main__":
    file_path = 'melo/data/example/metadata.list'
    old_text = '/content/drive/MyDrive/MeloTTS_Audios/'
    new_text = '/data/example/wavs/'

    replace_text_in_file(file_path, old_text, new_text)
    print(f"Text replacement complete in file: {file_path}")