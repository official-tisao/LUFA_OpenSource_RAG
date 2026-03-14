2026-01-03 13:21:32.279
PS C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG> conda activate
(base) PS C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG> conda deactivate   
PS C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG> conda activate LUFA_OpenSource_RAG
(lufa_rag) PS C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG> python -m venv venv
(lufa_rag) PS C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG> conda activate LUFA_OpenSource_RAG
(venv) (lufa_rag) PS C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG> pip install -r requirements.txt
Collecting llama-index==0.13.0
  Downloading llama_index-0.13.0-py3-none-any.whl (7.0 kB)
ERROR: Could not find a version that satisfies the requirement llama-index-llms-ollama==0.3.8 (from versions: 0.0.1, 0.1.0, 0.1.1, 0.1.2, 0.1.3, 0.1.4, 0.1.5, 0.
1.6, 0.2.0, 0.2.1, 0.2.2, 0.3.0, 0.3.1, 0.3.2, 0.3.3, 0.3.4, 0.3.5, 0.3.6, 0.4.0, 0.4.1, 0.4.2, 0.5.0, 0.5.1, 0.5.2, 0.5.3, 0.5.4, 0.5.5, 0.5.6, 0.6.0, 0.6.1, 0.6.2, 0.7.0, 0.7.1, 0.7.2, 0.7.3, 0.7.4, 0.8.0, 0.9.0, 0.9.1)
ERROR: No matching distribution found for llama-index-llms-ollama==0.3.8
(venv) (lufa_rag) PS C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG> # Make sure venv is active: (venv) in prompt
(venv) (lufa_rag) PS C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG> python src/ingestion.py
Starting document ingestion...
Loading English documents...
2026-02-14 09:19:00,922 - WARNING - Ignoring wrong pointing object 8 0 (offset 0)
2026-02-14 09:19:00,922 - WARNING - Ignoring wrong pointing object 10 0 (offset 0)
2026-02-14 09:19:00,922 - WARNING - Ignoring wrong pointing object 12 0 (offset 0)
2026-02-14 09:19:00,923 - WARNING - Ignoring wrong pointing object 20 0 (offset 0)
Loading French documents...
2026-02-14 09:21:27,687 - WARNING - incorrect startxref pointer(2)
2026-02-14 09:21:27,691 - WARNING - parsing for Object Streams
2026-02-14 09:21:27,864 - WARNING - Object 131 0 not defined.
2026-02-14 09:21:28,721 - WARNING - Error -3 while decompressing data: invalid distance too far back
Tagging documents with language...
Document tagged with language: en
Document tagged with language: fr
Document tagged with language: en
Initializing embedding model: bge-m3
Initializing ChromaDB at db/chroma_db...
2026-02-14 09:22:50,395 - INFO - Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
Creating vector store index...
Applying transformations: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.77s/it]
Generating embeddings:   0%|                                                                                                                              | 0/2048 [00:00<?, ?it/s]2026-02-14 09:23:10,812 - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 500 Internal Server Error"
Traceback (most recent call last):
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\src\ingestion.py", line 224, in <module>
    ingest_documents()
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\src\ingestion.py", line 217, in ingest_documents
    index = create_multilingual_index(english_dir, french_dir, db_path)
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\src\ingestion.py", line 137, in create_multilingual_index
    index = VectorStoreIndex.from_documents(
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\llama_index\core\indices\base.py", line 122, in from_documents
    return cls(
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\llama_index\core\indices\vector_store\base.py", line 75, in __init__
    super().__init__(
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\ollama\_client.py", line 133, in _request_raw
    raise ResponseError(e.response.text, e.response.status_code) from None
ollama._types.ResponseError: failed to encode response: json: unsupported value: NaN (status code: 500)
Generating embeddings:   0%|▌                                                                                                                     | 9/2048 [00:14<55:34,  1.64s/it]
(venv) (lufa_rag) PS C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG> python src/ingestion.py                             
Starting document ingestion...
Loading English documents...
2026-02-14 09:44:52,770 - WARNING - Ignoring wrong pointing object 8 0 (offset 0)
2026-02-14 09:44:52,770 - WARNING - Ignoring wrong pointing object 10 0 (offset 0)
2026-02-14 09:44:52,770 - WARNING - Ignoring wrong pointing object 12 0 (offset 0)
2026-02-14 09:44:52,770 - WARNING - Ignoring wrong pointing object 20 0 (offset 0)
2026-02-14 09:45:25,094 - WARNING - incorrect startxref pointer(2)
2026-02-14 09:45:25,094 - WARNING - parsing for Object Streams
2026-02-14 09:45:25,284 - WARNING - Object 131 0 not defined.
2026-02-14 09:45:26,202 - WARNING - Error -3 while decompressing data: invalid distance too far back
Traceback (most recent call last):
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\src\ingestion.py", line 224, in <module>
    ingest_documents()
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\src\ingestion.py", line 217, in ingest_documents
    index = create_multilingual_index(english_dir, french_dir, db_path)
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\src\ingestion.py", line 94, in create_multilingual_index
    english_docs = load_documents_from_directory(english_dir)
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\src\ingestion.py", line 40, in load_documents_from_directory
    documents = reader.load_data()
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\llama_index\core\readers\file\base.py", line 783, in load_data
    documents.extend(load_file_with_args(input_file))
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\llama_index\core\readers\file\base.py", line 613, in load_file
    docs = reader.load_data(input_file, **kwargs)
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\tenacity\__init__.py", line 331, in wrapped_f
    return copy(f, *args, **kw)
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\tenacity\__init__.py", line 470, in __call__
    do = self.iter(retry_state=retry_state)
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\tenacity\__init__.py", line 371, in iter
    result = action(retry_state)
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\tenacity\__init__.py", line 393, in <lambda>
    self._add_action_func(lambda rs: rs.outcome.result())
  File "C:\Users\TISAO-MSI\.conda\envs\lufa_rag\lib\concurrent\futures\_base.py", line 451, in result
    return self.__get_result()
  File "C:\Users\TISAO-MSI\.conda\envs\lufa_rag\lib\concurrent\futures\_base.py", line 403, in __get_result
    raise self._exception
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\tenacity\__init__.py", line 473, in __call__
    result = fn(*args, **kwargs)
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\llama_index\readers\file\docs\base.py", line 91, in load_data
    page_text = pdf.pages[page].extract_text()
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\pypdf\_page.py", line 2044, in extract_text
    return self._extract_text(
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\pypdf\_page.py", line 1739, in _extract_text
    for operands, operator in content.operations:
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\pypdf\generic\_data_structures.py", line 1412, in operations
    self._parse_content_stream(BytesIO(self._data))
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\pypdf\generic\_data_structures.py", line 1305, in _parse_content_stream
    operands.append(read_object(stream, None, self.forced_encoding))
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\pypdf\generic\_data_structures.py", line 1476, in read_object
    return NumberObject.read_from_stream(stream)
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\pypdf\generic\_base.py", line 588, in read_from_stream
    num = read_until_regex(stream, NumberObject.NumberPattern)
  File "C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG\venv\lib\site-packages\pypdf\_utils.py", line 259, in read_until_regex
    tok = stream.read(16)
KeyboardInterrupt
(venv) (lufa_rag) PS C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG> Remove-Item -Recurse -Force db/chroma_db
(venv) (lufa_rag) PS C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG> python src/ingestion.py                 
Starting document ingestion...
Loading English documents...
2026-02-14 09:46:57,302 - WARNING - Ignoring wrong pointing object 8 0 (offset 0)
2026-02-14 09:46:57,302 - WARNING - Ignoring wrong pointing object 10 0 (offset 0)
2026-02-14 09:46:57,302 - WARNING - Ignoring wrong pointing object 12 0 (offset 0)
2026-02-14 09:46:57,302 - WARNING - Ignoring wrong pointing object 20 0 (offset 0)
2026-02-14 09:47:27,228 - WARNING - incorrect startxref pointer(2)
2026-02-14 09:47:27,230 - WARNING - parsing for Object Streams
2026-02-14 09:47:27,396 - WARNING - Object 131 0 not defined.
2026-02-14 09:47:28,223 - WARNING - Error -3 while decompressing data: invalid distance too far back
Loading French documents...
2026-02-14 09:49:25,332 - WARNING - incorrect startxref pointer(2)
2026-02-14 09:49:25,338 - WARNING - parsing for Object Streams
2026-02-14 09:49:25,538 - WARNING - Object 131 0 not defined.
2026-02-14 09:49:26,459 - WARNING - Error -3 while decompressing data: invalid distance too far back
Tagging documents with language...
Document tagged with language: en
Document tagged with language: fr
Initializing embedding model: nomic-embed-text-v2-moe
Initializing ChromaDB at db/chroma_db...
2026-02-14 09:50:54,355 - INFO - Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
Creating vector store index...
Applying transformations: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.38s/it]
Generating embeddings:   0%|                                                                                                                              | 0/2048 [00:00<?, ?it/s]2026-02-14 09:51:08,325 - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
Generating embeddings:   0%|▌                                                                                                                    | 10/2048 [00:06<23:43,  1.43it/s]2026-02-14 09:51:10,487 - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
Generating embeddings: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2048/2048 [08:14<00:00,  4.14it/s]
Generating embeddings:   0%|                                                                                                                              | 0/2048 [00:00<?, ?it/s]2026-02-14 09:59:28,548 - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
Generating embeddings:   0%|▌                                                                                                                    | 10/2048 [00:04<16:52,  2.01it/s]2026-02-14 09:59:31,288 - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
Generating embeddings: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 2040/2048 [09:22<00:02,  3.21it/s]2026-02-14 10:08:48,598 - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
Generating embeddings: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2048/2048 [09:25<00:00,  3.62it/s]
Generating embeddings:   0%|                                                                                                                              | 0/1017 [00:00<?, ?it/s]2026-02-14 10:09:06,756 - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
Generating embeddings:   1%|█▏                                                                                                                   | 10/1017 [00:04<08:20,  2.01it/s]2026-02-14 10:09:09,493 - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
Generating embeddings:  99%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 1010/1017 [05:15<00:00,  7.77it/s]2026-02-14 10:14:18,392 - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
Generating embeddings: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1017/1017 [05:16<00:00,  3.21it/s]
Index created successfully with 3521 documents
Metadata updated: 2039 EN, 1482 FR documents
Document ingestion completed!
(venv) (lufa_rag) PS C:\Users\TISAO-MSI\Codebase\LUFA_OpenSource_RAG> streamlit run src/app.py

      Welcome to Streamlit!

      If you'd like to receive helpful onboarding emails, news, offers, promotions,
      and the occasional swag, please enter your email address below. Otherwise,
      leave this field blank.

      Email: stiamiyu@laurentian.ca                                                                                                                                                 

  You can find our privacy policy at https://streamlit.io/privacy-policy
  Summary:
  - This open source library collects usage statistics.
  - We cannot see and do not store information contained inside Streamlit apps,
    such as text, charts, images, etc.
  - Telemetry data is stored in servers in the United States.
  - If you'd like to opt out, add the following to %userprofile%/.streamlit/config.toml,
    creating that file if necessary:
    [browser]
    gatherUsageStats = false
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501                                                                                                                                                  
  Network URL: http://172.20.10.2:8501
Initializing LLM: llama3.2:3b-instruct-q4_K_M
Initializing embedding model: nomic-embed-text-v2-moe
Loading index from db/chroma_db...
Index loaded successfully
Detected query language: en
Detected query language: en
Detected query language: fr
Detected query language: fr
Detected query language: en
================================================================================
STARTING INTERACTIVE BILINGUAL PROCESSING LOOP
================================================================================

[1/212] Processing Question ID: test_en_001
   -> Query Preview: "What is the primary purpose of the collective agreement?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 61e470ca-8d46-4082-b958-76bd245c03be
      - Text Alignment Score: 80.95%

[2/212] Processing Question ID: test_en_002
   -> Query Preview: "Who is recognized as the exclusive bargaining agent for full-time..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 4b3dfb01-42c9-4d90-a9e3-df2a30fcb64e
      - Text Alignment Score: 83.33%

[3/212] Processing Question ID: test_en_003
   -> Query Preview: "Does the University of Sudbury collective agreement cover part-ti..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 4b3dfb01-42c9-4d90-a9e3-df2a30fcb64e
      - Text Alignment Score: 70.00%

[4/212] Processing Question ID: test_en_004
   -> Query Preview: "How is academic freedom defined?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: d16f73eb-6417-4570-b0e4-113213bf9199
      - Text Alignment Score: 100.00%

[5/212] Processing Question ID: test_en_005
   -> Query Preview: "Does academic freedom protect faculty from disciplinary action fo..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: c4372400-441b-4afa-ad53-784efbb6305b
      - Text Alignment Score: 77.78%

[6/212] Processing Question ID: test_en_006
   -> Query Preview: "What constitutes a grievance?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 19ccac9d-bb0a-45e7-88c9-cc4b0ded6f2e
      - Text Alignment Score: 89.47%

[7/212] Processing Question ID: test_en_007
   -> Query Preview: "What are the steps in the grievance procedure?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 1fcb921b-05f3-48e2-98d8-0f5cc306baa4
      - Text Alignment Score: 60.00%

[8/212] Processing Question ID: test_en_008
   -> Query Preview: "Can the Union file a policy grievance on behalf of multiple membe..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 3819b4da-d98a-46b3-afab-1056ff0d26ba
      - Text Alignment Score: 59.09%

[9/212] Processing Question ID: test_en_009
   -> Query Preview: "What is the role of an arbitrator?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 899b2ab9-ec3e-4459-ac89-777602721004
      - Text Alignment Score: 65.22%

[10/212] Processing Question ID: test_en_010
   -> Query Preview: "Is discrimination on the basis of political affiliation permitted..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: c45e2f80-a287-4e30-a0ab-c6b0b77d46ed
      - Text Alignment Score: 69.23%

[11/212] Processing Question ID: test_en_011
   -> Query Preview: "What grounds of discrimination are prohibited?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: f9398d48-bfc5-47ca-ad5b-b6b873887bc6
      - Text Alignment Score: 100.00%

[12/212] Processing Question ID: test_en_012
   -> Query Preview: "What is the standard normal teaching load for a tenure-track facu..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 26c9d4ca-c6af-4bbf-bf28-aafe0ddca6c8
      - Text Alignment Score: 73.91%

[13/212] Processing Question ID: test_en_013
   -> Query Preview: "How are workload reductions granted?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 824a311c-15f4-4618-8e4d-6867f9f43b61
      - Text Alignment Score: 56.25%

[14/212] Processing Question ID: test_en_014
   -> Query Preview: "Are faculty members required to participate in university governa..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 65260c5d-e54c-43e4-9c73-5d84c0b27817
      - Text Alignment Score: 68.18%

[15/212] Processing Question ID: test_en_015
   -> Query Preview: "What is the procedure for establishing a new faculty position?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: ca882712-f706-41af-8ea4-dd10f2f4878c
      - Text Alignment Score: 77.78%

[16/212] Processing Question ID: test_en_016
   -> Query Preview: "What are the ranks of faculty appointments?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 4aecc77e-ec5b-47d7-b667-2d900ba9976e
      - Text Alignment Score: 100.00%

[17/212] Processing Question ID: test_en_017
   -> Query Preview: "How long is the initial probationary period for a tenure-track As..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 4bc90158-eb5f-478c-934d-49a942035104
      - Text Alignment Score: 92.86%

[18/212] Processing Question ID: test_en_018
   -> Query Preview: "When must a faculty member apply for tenure?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 79d7f5e6-8771-49c3-9196-3fcfed3b19b6
      - Text Alignment Score: 80.95%

[19/212] Processing Question ID: test_en_019
   -> Query Preview: "What criteria are evaluated for granting tenure?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 913da199-a9b5-4399-b962-936dc893ee7b
      - Text Alignment Score: 64.71%

[20/212] Processing Question ID: test_en_020
   -> Query Preview: "Who assesses the tenure application initially?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 0b7aed9d-187d-4f48-930f-e08288984bd8
      - Text Alignment Score: 75.00%

[21/212] Processing Question ID: test_en_021
   -> Query Preview: "What happens if tenure is denied?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 79d7f5e6-8771-49c3-9196-3fcfed3b19b6
      - Text Alignment Score: 71.43%

[22/212] Processing Question ID: test_en_022
   -> Query Preview: "What is required for promotion to Full Professor?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: dc35f270-10bf-4ab1-ad3a-1d37a4e7f0c1
      - Text Alignment Score: 68.18%

[23/212] Processing Question ID: test_en_023
   -> Query Preview: "Can external referees be used in the promotion process?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 699fafce-3102-45a1-a8f9-5328252734c1
      - Text Alignment Score: 81.25%

[24/212] Processing Question ID: test_en_024
   -> Query Preview: "What is a Limited Term Appointment (LTA)?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: d1aa054a-d972-48d6-ada4-158f785f6f87
      - Text Alignment Score: 78.26%

[25/212] Processing Question ID: test_en_025
   -> Query Preview: "How are base salaries determined?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: c0074b4f-85e4-47d0-81c7-32cfda27a6a4
      - Text Alignment Score: 55.56%

[26/212] Processing Question ID: test_en_026
   -> Query Preview: "What is a Career Development Increment (CDI)?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: fc8d5dca-a890-487e-814f-932649528497
      - Text Alignment Score: 72.73%

[27/212] Processing Question ID: test_en_027
   -> Query Preview: "Are faculty members eligible for overload teaching pay?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 21a075fe-ff04-4647-92d6-e043649c8cd7
      - Text Alignment Score: 76.47%

[28/212] Processing Question ID: test_en_028
   -> Query Preview: "What happens to the salary grid under the mediated term sheet?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 30b4edc0-0d11-4ad4-a17a-6c332c922c09
      - Text Alignment Score: 50.00%

[29/212] Processing Question ID: test_en_029
   -> Query Preview: "Does the university provide health and dental benefits?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: e8c380f8-aa10-4671-a087-bf8d04c2a360
      - Text Alignment Score: 64.71%

[30/212] Processing Question ID: test_en_030
   -> Query Preview: "Is the pension plan a defined benefit or defined contribution pla..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 0c4edca6-fd9e-4752-b41a-7500cdc9f267
      - Text Alignment Score: 63.64%

[31/212] Processing Question ID: test_en_031
   -> Query Preview: "What is the professional development allowance (PER)?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 063ff2fc-51af-470f-9a14-8d7855fe047b
      - Text Alignment Score: 65.00%

[32/212] Processing Question ID: test_en_032
   -> Query Preview: "When is a faculty member eligible for a sabbatical leave?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: c30b573f-8d0f-4b3b-bde2-58ebdb6c585d
      - Text Alignment Score: 77.78%

[33/212] Processing Question ID: test_en_033
   -> Query Preview: "What percentage of salary is paid during a 12-month sabbatical?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 084adbcf-1a37-4840-b13f-0737bc0f6d86
      - Text Alignment Score: 65.22%

[34/212] Processing Question ID: test_en_034
   -> Query Preview: "Can a member take a 6-month sabbatical instead?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 315cdcf6-c4a0-4784-8619-b8ce4f3509bb
      - Text Alignment Score: 93.33%

[35/212] Processing Question ID: test_en_035
   -> Query Preview: "Does the collective agreement offer pregnancy/maternity leave?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 13673c42-5a60-4988-94d0-0a8328a18382
      - Text Alignment Score: 60.00%

[36/212] Processing Question ID: test_en_036
   -> Query Preview: "How long does the supplemental employment benefit (top-up) last f..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 83f74e21-6dce-4d7d-ae47-743b69b13b0e
      - Text Alignment Score: 65.38%

[37/212] Processing Question ID: test_en_037
   -> Query Preview: "Is there a provision for parental leave?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: df7f727b-1fe4-4fd8-8810-70b1d7b1ef40
      - Text Alignment Score: 55.56%

[38/212] Processing Question ID: test_en_038
   -> Query Preview: "What is the policy for sick leave?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a180e699-9165-4963-8a01-6d98ec4f1039
      - Text Alignment Score: 74.07%

[39/212] Processing Question ID: test_en_039
   -> Query Preview: "Are members entitled to bereavement leave?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: f870ecb1-be72-498a-9e7f-cc039597bc0d
      - Text Alignment Score: 73.68%

[40/212] Processing Question ID: test_en_040
   -> Query Preview: "Can a faculty member take an unpaid leave of absence?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 62f1491c-ac80-4014-87cf-4818cf2c723e
      - Text Alignment Score: 63.16%

[41/212] Processing Question ID: test_en_041
   -> Query Preview: "What constitutes 'just cause' for discipline?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 82566dfe-76f4-4e19-9bd0-b29aaf1ae6de
      - Text Alignment Score: 77.27%

[42/212] Processing Question ID: test_en_042
   -> Query Preview: "Does the member have the right to union representation during dis..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 74354667-4720-4945-a143-dfb2ec8fb5b1
      - Text Alignment Score: 76.19%

[43/212] Processing Question ID: test_en_043
   -> Query Preview: "What is progressive discipline?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 8d9ff986-72dc-410f-acc7-83e715af930c
      - Text Alignment Score: 47.83%

[44/212] Processing Question ID: test_en_044
   -> Query Preview: "How are intellectual property rights managed?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 5a594c86-4c00-406f-8dc2-f3eafa2d65f8
      - Text Alignment Score: 56.52%

[45/212] Processing Question ID: test_en_045
   -> Query Preview: "Who owns the patent if a member creates a patentable invention us..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 7f271b5a-3162-4c3a-845c-91a37e3e4617
      - Text Alignment Score: 63.64%

[46/212] Processing Question ID: test_en_046
   -> Query Preview: "Are faculty members evaluated on their teaching?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
