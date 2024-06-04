import os
import sys
import pandas as pd
import json
import urllib.request


def create_directory(dir_path):
    """
    Create a directory using the specified path.

    Parameters:
    dir_path (str): Path of the directory to be created.

    Raises:
    FileExistsError: If the directory already exists.
    """
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        print(f'Directory "{dir_path}" exists.', file=sys.stderr)


def delete_directory_contents(dir_path):
    """
    Delete all contents of a specified directory.

    Parameters:
    dir_path (str): Path of the directory where the contents need to be deleted.

    Raises:
    OSError: If an error occurs while deleting files or directories.
    """
    try:
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
            for name in dirs:
                dir_path = os.path.join(root, name)
                os.rmdir(dir_path)
    except OSError:
        print('Error encountered while deleting files.', file=sys.stderr)


def generate_files(ids, sample_size=100, seed=1, max_papers=5):
    """
    Generate discussion section files and summary JSON files for a given list of paper IDs.

    Parameters:
    ids (list): List of paper IDs to process.
    sample_size (int, optional): Number of papers to sample. Default is 100.
    seed (int, optional): Random seed for reproducibility. Default is 1.
    max_papers (int, optional): Maximum number of papers to process. Default is 5.

    Raises:
    Exception: If any error occurs during the creation of the files.
    """
    # Create a directory for storing discussion sections for each paper
    dir_path = './data_construction/'
    dir_name = dir_path + 'discussions/'
    dir_summ = dir_path + 'summaries_json/'
    create_directory(dir_path)
    create_directory(dir_name)
    delete_directory_contents(dir_name)

    create_directory(dir_summ)
    delete_directory_contents(dir_summ)

    n = 0

    for id in ids:
        json_url = f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{id}/unicode'
        article_url = f'https://www.ncbi.nlm.nih.gov/pmc/articles/{id}/'

        with urllib.request.urlopen(json_url) as article:
            article_data = json.load(article)[0]
            discussion_text = ''

            # Extract the text from the discussion sections
            passages = article_data['documents'][0]['passages']
            for passage in passages:
                if passage['infons'].get('section_type') == 'DISCUSS' and passage['infons'].get('type') == 'paragraph':
                    discussion_text += passage['text'] + '\n'

            # Write Discussion section to file
            if discussion_text:
                f = open(f'{dir_name}/{id}.txt', 'w', encoding='utf-8')
                f.write(discussion_text)

                # Create JSON files for each ID
                summ_dict = dict()
                for model in ['ChatGPT', 'Gemini', 'MSCopilot']:
                    summ_dict[model] = {
                        "summ": "",
                        "quality": {"Factual consistency": [0, ""],
                                    "Coverage": [0, ""],
                                    "Coherence": [0, ""],},
                        "rank": 0
                    }
                f = open(f'{dir_summ}/{id}.json', 'w', encoding='utf-8')
                json.dump(summ_dict, f, indent=4)
                n += 1
                print(f'{n} --- {id} --- Done. --- ({article_url})')
                if n >= max_papers:
                    break
    print(f'Succesfully created files.')


def main():

    # CSV file containing PubMed Central Accession ID's
    # Downloaded from: https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/
    df = pd.read_csv('oa_comm_xml.PMC010xxxxxx.baseline.2023-12-18.filelist.csv')

    # Settings to ensure reproducability
    sample_size = 300
    seed = 1
    max_papers = 15

    accession_ids = df[df['AccessionID'] != 0].sample(n=sample_size, random_state=seed)['AccessionID']

    generate_files(accession_ids, sample_size, seed, max_papers)


if __name__=="__main__":
    main()
 