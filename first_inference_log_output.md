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
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: e42aef3f-86cd-487d-9417-16745ecca0c4
      - Text Alignment Score: 57.14%

[47/212] Processing Question ID: test_en_047
   -> Query Preview: "Can student evaluations be the sole metric for teaching performan..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 003e2ac8-a452-462d-bc19-815654b44e7f
      - Text Alignment Score: 57.89%

[48/212] Processing Question ID: test_en_048
   -> Query Preview: "What is the role of a Department Chair?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 161bee54-822d-43c3-883f-9f8af03890b0
      - Text Alignment Score: 68.00%

[49/212] Processing Question ID: test_en_049
   -> Query Preview: "How is a Department Chair selected?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 9c7b3ebd-9b6e-4b27-9d89-f2a0d6646871
      - Text Alignment Score: 80.00%

[50/212] Processing Question ID: test_en_050
   -> Query Preview: "Do Chairs receive additional compensation?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 2a5b8f28-4715-4caf-a691-1ea8dd2031a5
      - Text Alignment Score: 85.71%

[51/212] Processing Question ID: test_en_051
   -> Query Preview: "Is there a retirement age specified?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 34e5e5f2-ad6a-4d07-a80e-a6a2677e1a6d
      - Text Alignment Score: 42.86%

[52/212] Processing Question ID: test_en_052
   -> Query Preview: "Are phased retirement options available?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: c08d9013-e02c-475c-88f6-46e9280f9546
      - Text Alignment Score: 68.18%

[53/212] Processing Question ID: test_en_053
   -> Query Preview: "What happens in the event of financial exigency?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: e6ba4dbc-ca5f-4cdb-9238-437b65de90f9
      - Text Alignment Score: 47.62%

[54/212] Processing Question ID: test_en_054
   -> Query Preview: "How did the CCAA mediation term sheet affect financial exigency c..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: d4a59e09-d098-4089-a8e5-a5837a1af91d
      - Text Alignment Score: 61.90%

[55/212] Processing Question ID: test_en_055
   -> Query Preview: "Are faculty members allowed to engage in outside professional act..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: e602ea85-1a03-48d0-9fa1-be00006114d5
      - Text Alignment Score: 76.00%

[56/212] Processing Question ID: test_en_056
   -> Query Preview: "Do members need to report their outside professional activities?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 3c319deb-2715-4832-9769-3c613b12198f
      - Text Alignment Score: 68.42%

[57/212] Processing Question ID: test_en_057
   -> Query Preview: "Is health and safety the responsibility of the university?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 987885ed-fa92-4fc0-84d6-5f2e5d2b05a0
      - Text Alignment Score: 88.24%

[58/212] Processing Question ID: test_en_058
   -> Query Preview: "Is there a Joint Health and Safety Committee?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 171dcace-aceb-4fcf-9d02-31e8d76340a9
      - Text Alignment Score: 66.67%

[59/212] Processing Question ID: test_en_059
   -> Query Preview: "What access do members have to their personnel files?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 8d1d21eb-fb69-45a7-b991-20290cd53291
      - Text Alignment Score: 86.36%

[60/212] Processing Question ID: test_en_060
   -> Query Preview: "Can anonymous complaints be added to a member's personnel file?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 4ba66300-c277-4528-b968-d481218d9279
      - Text Alignment Score: 80.77%

[61/212] Processing Question ID: test_en_061
   -> Query Preview: "How is the distance education or online teaching workload determi..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 9686d26e-f310-4aaa-814e-afb40eea8111
      - Text Alignment Score: 52.94%

[62/212] Processing Question ID: test_en_062
   -> Query Preview: "Is technical support guaranteed for online courses?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 0fa01f7f-2c0e-472d-a02a-b923dc4d80ec
      - Text Alignment Score: 58.82%

[63/212] Processing Question ID: test_en_063
   -> Query Preview: "What happens if a scheduled class is cancelled due to low enrollm..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a6d1f4c5-5890-4639-a7a5-6ae4ce5aa9a5
      - Text Alignment Score: 86.96%

[64/212] Processing Question ID: test_en_064
   -> Query Preview: "Are faculty members expected to be on campus during the summer te..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a5b1681d-616b-4333-87d1-8703d0694b23
      - Text Alignment Score: 66.67%

[65/212] Processing Question ID: test_en_065
   -> Query Preview: "How many vacation days do faculty members receive?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 1ae30de3-c792-4b47-a6d3-6147f1740182
      - Text Alignment Score: 70.00%

[66/212] Processing Question ID: test_en_066
   -> Query Preview: "Is it possible to carry forward unused vacation days?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 1ae30de3-c792-4b47-a6d3-6147f1740182
      - Text Alignment Score: 57.89%

[67/212] Processing Question ID: test_en_067
   -> Query Preview: "What is the Joint Committee on the Administration of the Agreemen..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: bf22c193-0d07-4534-805e-3a7c2423c7f5
      - Text Alignment Score: 71.43%

[68/212] Processing Question ID: test_en_068
   -> Query Preview: "How are union dues collected?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 2b2b257e-afed-40a2-87ba-9cf4e7568bdf
      - Text Alignment Score: 61.90%

[69/212] Processing Question ID: test_en_069
   -> Query Preview: "Does the Union get space on campus?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: d728f6cb-1903-46a3-a753-35f21b58ff5e
      - Text Alignment Score: 70.59%

[70/212] Processing Question ID: test_en_070
   -> Query Preview: "What happens if a holiday falls on a member's scheduled vacation?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 1ae30de3-c792-4b47-a6d3-6147f1740182
      - Text Alignment Score: 73.33%

[71/212] Processing Question ID: test_en_071
   -> Query Preview: "Is there a provision for court leave?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 266f2f22-f266-4f60-957f-b9be903d0345
      - Text Alignment Score: 75.00%

[72/212] Processing Question ID: test_en_072
   -> Query Preview: "How is a complaint of harassment handled?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 75062832-f038-47c8-a417-0debdd8dadf8
      - Text Alignment Score: 66.67%

[73/212] Processing Question ID: test_en_073
   -> Query Preview: "Can the Union access information about the university's financial..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: b1030a7d-24c7-4d40-9c97-1faa03dd017d
      - Text Alignment Score: 63.16%

[74/212] Processing Question ID: test_en_074
   -> Query Preview: "What was the context of the LUFA Term Sheet?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 7f474aab-bfdb-4f69-9416-f8264530f2b2
      - Text Alignment Score: 55.56%

[75/212] Processing Question ID: test_en_075
   -> Query Preview: "Are there specific provisions for Indigenous faculty?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: c260d201-d60a-4c55-a254-869c3b1449fc
      - Text Alignment Score: 50.00%

[76/212] Processing Question ID: test_en_076
   -> Query Preview: "Can a faculty member be forced to move to a different campus?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 112d77d8-6448-4d40-af69-cb344b3b3f1e
      - Text Alignment Score: 76.47%

[77/212] Processing Question ID: test_en_077
   -> Query Preview: "Who defines the curriculum for a degree program?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: f05016c5-7d90-415c-8cd3-9211f1811baf
      - Text Alignment Score: 71.43%

[78/212] Processing Question ID: test_en_078
   -> Query Preview: "Does the employer provide moving expenses for new faculty?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 47358428-1f1e-4a10-a217-72654c96fa96
      - Text Alignment Score: 85.71%

[79/212] Processing Question ID: test_en_079
   -> Query Preview: "What is a bipartite appointment?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: ca6c15c5-f91f-4e21-b88b-a18de616cb73
      - Text Alignment Score: 52.63%

[80/212] Processing Question ID: test_en_080
   -> Query Preview: "What is a tripartite appointment?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: e2ff997f-3679-455f-aed8-31d8c70bb227
      - Text Alignment Score: 55.56%

[81/212] Processing Question ID: test_en_081
   -> Query Preview: "Can a bipartite member transition to a tripartite appointment?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 767139db-6759-4635-8142-c18f45b31cfc
      - Text Alignment Score: 56.25%

[82/212] Processing Question ID: test_en_082
   -> Query Preview: "How are cross-appointments handled?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 8672b8bb-fbf1-483b-8a28-af28cb93f3bb
      - Text Alignment Score: 73.68%

[83/212] Processing Question ID: test_en_083
   -> Query Preview: "Does the collective agreement protect whistleblower activity?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 13b50482-9f00-41d6-80ee-12512d52e416
      - Text Alignment Score: 42.86%

[84/212] Processing Question ID: test_en_084
   -> Query Preview: "How is a strike or lockout resolved under the collective agreemen..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a100e425-7f46-41a4-b3de-529a011bd855
      - Text Alignment Score: 75.00%

[85/212] Processing Question ID: test_en_085
   -> Query Preview: "Can the university change the pension plan without union consent?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 504521f0-19a0-469b-b2ba-4c11144e87dc
      - Text Alignment Score: 82.35%

[86/212] Processing Question ID: test_en_086
   -> Query Preview: "Is the University of Sudbury collective agreement separate from L..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: fbcdd1a3-c266-404c-821a-c69ad009d18b
      - Text Alignment Score: 58.82%

[87/212] Processing Question ID: test_en_087
   -> Query Preview: "Are tuition waivers provided for dependents?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: dbe7d39c-7050-4cbf-981d-e055c7cdd2b7
      - Text Alignment Score: 73.91%

[88/212] Processing Question ID: test_en_088
   -> Query Preview: "What happens if a member resigns?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 34e5e5f2-ad6a-4d07-a80e-a6a2677e1a6d
      - Text Alignment Score: 86.36%

[89/212] Processing Question ID: test_en_089
   -> Query Preview: "How is the performance of a faculty member who is on sick leave e..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a180e699-9165-4963-8a01-6d98ec4f1039
      - Text Alignment Score: 66.67%

[90/212] Processing Question ID: test_en_090
   -> Query Preview: "What is a 'Member of the Bargaining Unit'?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 3aae6285-da2b-4197-98f7-f8b4e64c1084
      - Text Alignment Score: 73.91%

[91/212] Processing Question ID: test_en_091
   -> Query Preview: "What role does the Senate play in collective bargaining?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a5c57fe7-b731-40ba-8c62-ab7b2b68f2b6
      - Text Alignment Score: 71.43%

[92/212] Processing Question ID: test_en_092
   -> Query Preview: "Can management assign a course to a member outside their area of ..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 1a6da860-3c54-4b70-a882-c46d1769ecd7
      - Text Alignment Score: 77.78%

[93/212] Processing Question ID: test_en_093
   -> Query Preview: "Are library faculty covered by this agreement?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 791425fe-85a9-417a-808d-1e38120cfce7
      - Text Alignment Score: 60.00%

[94/212] Processing Question ID: test_en_094
   -> Query Preview: "What is the probationary period for a professional librarian?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: d1aa054a-d972-48d6-ada4-158f785f6f87
      - Text Alignment Score: 57.89%

[95/212] Processing Question ID: test_en_095
   -> Query Preview: "Does the term sheet affect pension contributions?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 87744ec4-f20f-4e03-a1a6-baba02384508
      - Text Alignment Score: 60.00%

[96/212] Processing Question ID: test_en_096
   -> Query Preview: "Can the university hire non-union contractors to perform faculty ..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 28c7a7a1-0999-45df-bc69-1001010c29fd
      - Text Alignment Score: 68.42%

[97/212] Processing Question ID: test_en_097
   -> Query Preview: "What happens if a conflict arises between university policy and t..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 2ee165a3-4683-4f58-8534-6043810212a2
      - Text Alignment Score: 91.67%

[98/212] Processing Question ID: test_en_098
   -> Query Preview: "Is the university required to accommodate members with disabiliti..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 4504171f-ce45-4f3b-b005-6629f98f9569
      - Text Alignment Score: 52.63%

[99/212] Processing Question ID: test_en_099
   -> Query Preview: "Are faculty members evaluated on community service?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: e97de394-168a-4b71-9912-9bffab519981
      - Text Alignment Score: 61.90%

[100/212] Processing Question ID: test_en_100
   -> Query Preview: "Does the agreement regulate the use of university email?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: e3305383-7a57-440e-b211-b0e228444314
      - Text Alignment Score: 40.91%

[101/212] Processing Question ID: test_en_101
   -> Query Preview: "What is the duration of the LUFA 2017-2020 agreement?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: d32bfaa0-1e6c-4962-ab3d-fd0986ba51d7
      - Text Alignment Score: 76.19%

[102/212] Processing Question ID: test_en_102
   -> Query Preview: "What are the office hours requirements for faculty members?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 8f80464d-5daa-427b-b1fe-e2648e9cf02d
      - Text Alignment Score: 75.00%

[103/212] Processing Question ID: test_en_103
   -> Query Preview: "What is the policy on academic freedom?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: aac5ed3c-011d-4c64-839f-48bab0f03320
      - Text Alignment Score: 87.50%

[104/212] Processing Question ID: test_en_104
   -> Query Preview: "How is teaching load calculated?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 74f7eb5d-c451-487c-8505-221d5c3d7187
      - Text Alignment Score: 72.73%

[105/212] Processing Question ID: test_en_105
   -> Query Preview: "What are the provisions for sabbatical leave?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: c30b573f-8d0f-4b3b-bde2-58ebdb6c585d
      - Text Alignment Score: 85.71%

[106/212] Processing Question ID: test_en_106
   -> Query Preview: "What is the grievance procedure?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'en'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 26fb69f1-4da5-4f43-ab18-79335c16109a
      - Text Alignment Score: 69.23%

[107/212] Processing Question ID: test_fr_001
   -> Query Preview: "Quel est le but principal de la convention collective ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: b78eaafc-83f9-4c24-9081-cf23278df152
      - Text Alignment Score: 81.82%

[108/212] Processing Question ID: test_fr_002
   -> Query Preview: "Qui est reconnu comme l'agent négociateur exclusif des professeur..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: d37189e8-c9ff-46f9-a57e-7524cfc8e50d
      - Text Alignment Score: 75.00%

[109/212] Processing Question ID: test_fr_003
   -> Query Preview: "Les droits de la direction sont-ils soumis à la convention collec..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 6e5d1959-06f2-4afb-a7b5-bdb917aa467c
      - Text Alignment Score: 68.75%

[110/212] Processing Question ID: test_fr_004
   -> Query Preview: "Comment la liberté universitaire est-elle définie ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 164cf36f-82d1-4e6f-a8ee-861f619eba3b
      - Text Alignment Score: 87.50%

[111/212] Processing Question ID: test_fr_005
   -> Query Preview: "La liberté universitaire protège-t-elle les professeurs dans leur..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 9704f595-2abb-4f60-9b54-aced36aaef27
      - Text Alignment Score: 80.00%

[112/212] Processing Question ID: test_fr_006
   -> Query Preview: "Quels sont les motifs de discrimination interdits ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: af86d12a-15d9-4690-97fa-a4568ed696dd
      - Text Alignment Score: 78.26%

[113/212] Processing Question ID: test_fr_007
   -> Query Preview: "Est-il permis de discriminer un membre en raison de son affiliati..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: e56b0318-14af-4c74-b437-ffa371bce247
      - Text Alignment Score: 64.29%

[114/212] Processing Question ID: test_fr_008
   -> Query Preview: "L'Université a-t-elle le devoir d'accommoder les membres handicap..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: e3cb6d65-063d-4687-831c-93f7543337d0
      - Text Alignment Score: 59.09%

[115/212] Processing Question ID: test_fr_009
   -> Query Preview: "Comment le harcèlement est-il traité ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 0c51f02c-b3de-4f93-a376-1182a37b4c3f
      - Text Alignment Score: 54.55%

[116/212] Processing Question ID: test_fr_010
   -> Query Preview: "Qu'est-ce qu'un grief ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 74e1fffd-55a5-42ba-a768-ea014fe0783c
      - Text Alignment Score: 88.89%

[117/212] Processing Question ID: test_fr_011
   -> Query Preview: "Quelles sont les étapes formelles de la procédure de grief ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 22b46d8d-149e-4c61-940b-03052399e488
      - Text Alignment Score: 50.00%

[118/212] Processing Question ID: test_fr_012
   -> Query Preview: "L'APPUL peut-elle déposer un grief de principe ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: dd155bb5-4982-4e5e-9dd4-eba41c427b94
      - Text Alignment Score: 69.57%

[119/212] Processing Question ID: test_fr_013
   -> Query Preview: "Quel est le rôle d'un arbitre ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 4c91bdd2-a157-4f63-afea-b47f2be09c03
      - Text Alignment Score: 66.67%

[120/212] Processing Question ID: test_fr_014
   -> Query Preview: "Comment l'APPUL perçoit-elle les cotisations syndicales ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: ca5b39bb-ea8a-426f-8257-3c9f603fdaa8
      - Text Alignment Score: 65.00%

[121/212] Processing Question ID: test_fr_015
   -> Query Preview: "L'APPUL a-t-elle droit à un bureau sur le campus ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: b28185a2-0d07-43d8-9bee-0404f8e47e75
      - Text Alignment Score: 61.11%

[122/212] Processing Question ID: test_fr_016
   -> Query Preview: "Qu'est-ce que le Comité paritaire (JCAA) ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 207490ee-984b-49db-82dc-7f9e641c1eca
      - Text Alignment Score: 71.43%

[123/212] Processing Question ID: test_fr_017
   -> Query Preview: "Y a-t-il un droit de grève pendant la convention ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 5f16de46-02c3-48fe-b746-a6ce7bfbd704
      - Text Alignment Score: 57.89%

[124/212] Processing Question ID: test_fr_018
   -> Query Preview: "Les professeurs peuvent-ils consulter leur dossier personnel ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a070ebaf-d448-4911-b10e-ee671e89665b
      - Text Alignment Score: 66.67%

[125/212] Processing Question ID: test_fr_019
   -> Query Preview: "Les plaintes anonymes peuvent-elles être ajoutées au dossier pers..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a070ebaf-d448-4911-b10e-ee671e89665b
      - Text Alignment Score: 76.92%

[126/212] Processing Question ID: test_fr_020
   -> Query Preview: "Un membre peut-il ajouter des commentaires à son dossier personne..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: da6733ea-e196-4a9f-82c4-cb7a26ee1969
      - Text Alignment Score: 80.00%

[127/212] Processing Question ID: test_fr_021
   -> Query Preview: "Quels sont les rangs pour les professeurs ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 400a05e3-2b6a-44b0-b4d3-0455af004acf
      - Text Alignment Score: 91.67%

[128/212] Processing Question ID: test_fr_022
   -> Query Preview: "Qu'est-ce qu'une nomination à durée déterminée (LTA) ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: bbfba659-b2c0-4f6f-97e3-da8b2cfd91e0
      - Text Alignment Score: 73.68%

[129/212] Processing Question ID: test_fr_023
   -> Query Preview: "Quelle est la durée de la période d'essai pour un professeur adjo..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 84bf8188-bc95-492b-b764-8d65841769dc
      - Text Alignment Score: 82.35%

[130/212] Processing Question ID: test_fr_024
   -> Query Preview: "Quand un membre doit-il demander la permanence ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: aaaba503-4916-4777-883b-79b47eba6196
      - Text Alignment Score: 63.16%

[131/212] Processing Question ID: test_fr_025
   -> Query Preview: "Quels sont les critères pour obtenir la permanence ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: aaaba503-4916-4777-883b-79b47eba6196
      - Text Alignment Score: 63.16%

[132/212] Processing Question ID: test_fr_026
   -> Query Preview: "Qui fait l'évaluation initiale pour la permanence ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: c9bc559e-1ef0-4ed4-962b-9c07b53f8dd9
      - Text Alignment Score: 85.71%

[133/212] Processing Question ID: test_fr_027
   -> Query Preview: "Que se passe-t-il si la permanence est refusée ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: f2ff4aba-ce55-4cb6-b2b0-42002e7cee72
      - Text Alignment Score: 73.33%

[134/212] Processing Question ID: test_fr_028
   -> Query Preview: "Quelles sont les conditions pour la promotion au rang de professe..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 23ce4209-06b4-4cf0-acb6-443723ff8ffe
      - Text Alignment Score: 61.90%

[135/212] Processing Question ID: test_fr_029
   -> Query Preview: "Des évaluateurs externes sont-ils requis pour les promotions ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 92a32891-645c-4a74-947e-d738253b8440
      - Text Alignment Score: 65.00%

[136/212] Processing Question ID: test_fr_030
   -> Query Preview: "Qu'est-ce qu'une nomination bipartite ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 5f52aa99-13ae-47ea-86bb-81413164e4a7
      - Text Alignment Score: 68.18%

[137/212] Processing Question ID: test_fr_031
   -> Query Preview: "Qu'est-ce qu'une nomination tripartite ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 5f52aa99-13ae-47ea-86bb-81413164e4a7
      - Text Alignment Score: 71.43%

[138/212] Processing Question ID: test_fr_032
   -> Query Preview: "Un membre bipartite peut-il devenir tripartite ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: e9635588-5ca8-4e02-a05c-a291b16ec8c9
      - Text Alignment Score: 64.29%

[139/212] Processing Question ID: test_fr_033
   -> Query Preview: "Quelle est la charge normale d'enseignement pour un professeur me..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 646f3643-7525-4f95-9607-21237a0b31ed
      - Text Alignment Score: 78.95%

[140/212] Processing Question ID: test_fr_034
   -> Query Preview: "Les professeurs doivent-ils participer à la gouvernance de l'univ..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 417e2b1e-6b2a-4fea-9714-545843d3fcc3
      - Text Alignment Score: 61.11%

[141/212] Processing Question ID: test_fr_035
   -> Query Preview: "Comment obtenir une réduction de la charge de travail ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 6b5451ac-1538-4a33-897e-78b697f7f915
      - Text Alignment Score: 58.82%

[142/212] Processing Question ID: test_fr_036
   -> Query Preview: "La direction peut-elle forcer un membre à enseigner un cours hors..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: f422f3c5-f12f-4762-b24a-7c1d62062ecf
      - Text Alignment Score: 64.71%

[143/212] Processing Question ID: test_fr_037
   -> Query Preview: "Comment la charge pour les cours en ligne est-elle calculée ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: abc12e11-6888-4489-9787-cf486041f1b6
      - Text Alignment Score: 78.95%

[144/212] Processing Question ID: test_fr_038
   -> Query Preview: "Que se passe-t-il si un cours est annulé pour manque d'étudiants ..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: ddedbf67-7d4b-4e3b-a369-a0a1e0789d92
      - Text Alignment Score: 86.96%

[145/212] Processing Question ID: test_fr_039
   -> Query Preview: "Les membres doivent-ils être sur le campus l'été ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: e3cb6d65-063d-4687-831c-93f7543337d0
      - Text Alignment Score: 80.77%

[146/212] Processing Question ID: test_fr_040
   -> Query Preview: "Quelles sont les obligations concernant les heures de bureau ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 72e584ce-d52c-4084-b589-18c250a3b6f8
      - Text Alignment Score: 66.67%

[147/212] Processing Question ID: test_fr_041
   -> Query Preview: "L'employeur garantit-il le soutien technique pour l'enseignement ..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 69e7c14f-93f6-4b43-ad9f-2ba23cf98159
      - Text Alignment Score: 57.89%

[148/212] Processing Question ID: test_fr_042
   -> Query Preview: "Les courriels des professeurs sont-ils surveillés ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: caa161ac-46b5-4488-9ca7-4d7819f2add7
      - Text Alignment Score: 47.37%

[149/212] Processing Question ID: test_fr_043
   -> Query Preview: "Un membre peut-il être transféré d'un campus à l'autre ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 3598be29-4ec7-40e1-998e-fe2caccf3e74
      - Text Alignment Score: 56.25%

[150/212] Processing Question ID: test_fr_044
   -> Query Preview: "Qu'est-ce qui constitue une cause juste pour une mesure disciplin..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 363675f6-5d04-42ad-8431-9103552e07b2
      - Text Alignment Score: 61.11%

[151/212] Processing Question ID: test_fr_045
   -> Query Preview: "Le syndicat doit-il être présent lors d'une rencontre disciplinai..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a9375866-c4bf-449c-b916-2834b91f4070
      - Text Alignment Score: 71.43%

[152/212] Processing Question ID: test_fr_046
   -> Query Preview: "Qu'est-ce que la discipline progressive ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: de2297eb-56d3-4b46-b0da-dcc1539d94f5
      - Text Alignment Score: 56.52%

[153/212] Processing Question ID: test_fr_047
   -> Query Preview: "Les professeurs sont-ils évalués sur leur enseignement ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 70f4f298-2dd6-4a78-b33d-558bfa579611
      - Text Alignment Score: 62.50%

[154/212] Processing Question ID: test_fr_048
   -> Query Preview: "Les évaluations étudiantes sont-elles le seul critère pour juger ..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 2f223259-898a-4003-8bbd-596e0548a15f
      - Text Alignment Score: 58.82%

[155/212] Processing Question ID: test_fr_049
   -> Query Preview: "Comment le travail communautaire est-il évalué ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: c1734171-3d86-4d41-bc2f-4b562bb87e0d
      - Text Alignment Score: 68.42%

[156/212] Processing Question ID: test_fr_050
   -> Query Preview: "Comment un membre en congé de maladie est-il évalué pour la perma..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 38f8eac0-2a97-4a2a-8730-7ed2a42b91cf
      - Text Alignment Score: 75.00%

[157/212] Processing Question ID: test_fr_051
   -> Query Preview: "Comment les salaires de base sont-ils établis ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: acb35873-0399-4154-a937-d128e783ac04
      - Text Alignment Score: 50.00%

[158/212] Processing Question ID: test_fr_052
   -> Query Preview: "Qu'est-ce que l'Indemnité de développement de carrière (IDC) ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 0f170d44-be78-4372-8d34-a2dd1e5544fd
      - Text Alignment Score: 58.33%

[159/212] Processing Question ID: test_fr_053
   -> Query Preview: "Les membres reçoivent-ils une paie pour les cours en surcharge ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: df974ab9-ddfc-4642-b90c-92268186cdb0
      - Text Alignment Score: 84.21%

[160/212] Processing Question ID: test_fr_054
   -> Query Preview: "Comment la restructuration (Loi LACC) a-t-elle affecté les salair..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 2a135a64-092a-4cb8-a1c9-e844da7538df
      - Text Alignment Score: 50.00%

[161/212] Processing Question ID: test_fr_055
   -> Query Preview: "L'université offre-t-elle une assurance santé ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a3b9b830-3626-4484-bb50-5bd87b9a38f1
      - Text Alignment Score: 65.00%

[162/212] Processing Question ID: test_fr_056
   -> Query Preview: "Y a-t-il une allocation pour le perfectionnement professionnel (P..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a9a8ce4e-5673-458c-be1a-16a910d3f04c
      - Text Alignment Score: 81.82%

[163/212] Processing Question ID: test_fr_057
   -> Query Preview: "Les frais de scolarité des enfants des membres sont-ils couverts ..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: cd3add50-8ce7-4056-8b18-b0bb008e6268
      - Text Alignment Score: 86.96%

[164/212] Processing Question ID: test_fr_058
   -> Query Preview: "L'université peut-elle modifier le régime de retraite unilatérale..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 82402597-134a-403e-bbb0-a36d292418c7
      - Text Alignment Score: 57.89%

[165/212] Processing Question ID: test_fr_059
   -> Query Preview: "Le contrat imposé a-t-il affecté les cotisations de retraite ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: d60d66f0-2da6-4ae4-b63f-807dfc479b55
      - Text Alignment Score: 60.87%

[166/212] Processing Question ID: test_fr_060
   -> Query Preview: "Quand un membre a-t-il droit à un congé sabbatique de 12 mois ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: bfd80628-e1d3-479b-aad3-7ec37e8986aa
      - Text Alignment Score: 68.42%

[167/212] Processing Question ID: test_fr_061
   -> Query Preview: "Quelle est la rémunération pendant un congé sabbatique de 12 mois..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: f25b8f2c-44a7-4a03-8d25-f93a4b6d1162
      - Text Alignment Score: 50.00%

[168/212] Processing Question ID: test_fr_062
   -> Query Preview: "Peut-on prendre un congé sabbatique de 6 mois ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 91b559f8-d06f-4b46-ab8f-556d8d3de150
      - Text Alignment Score: 64.71%

[169/212] Processing Question ID: test_fr_063
   -> Query Preview: "Y a-t-il des dispositions pour le congé de maternité ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: f6cddf9e-93aa-4924-84fb-381bd8bde4d6
      - Text Alignment Score: 85.00%

[170/212] Processing Question ID: test_fr_064
   -> Query Preview: "Comment fonctionne la prestation supplémentaire (top-up) pour le ..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: d5299b85-1222-4bf3-9eeb-2ead7ca0ffd1
      - Text Alignment Score: 70.37%

[171/212] Processing Question ID: test_fr_065
   -> Query Preview: "Le congé parental est-il prévu dans l'entente ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 1cbf3b6d-bf46-4d3c-8760-2b5a42442509
      - Text Alignment Score: 63.16%

[172/212] Processing Question ID: test_fr_066
   -> Query Preview: "Quelle est la politique pour les congés de maladie ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 0d13cbc0-73b5-47c1-a3d5-9b27713c9ad0
      - Text Alignment Score: 71.43%

[173/212] Processing Question ID: test_fr_067
   -> Query Preview: "Les membres ont-ils droit à un congé de deuil ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: f7627ea2-bf3d-4a73-bf65-ac44567b8e4f
      - Text Alignment Score: 68.42%

[174/212] Processing Question ID: test_fr_068
   -> Query Preview: "Un professeur peut-il obtenir un congé sans solde ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: bee60054-a44f-45d7-acf2-1d409f6bc3b7
      - Text Alignment Score: 55.56%

[175/212] Processing Question ID: test_fr_069
   -> Query Preview: "Combien de jours de vacances les professeurs ont-ils ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: ebdfc320-0f1b-4421-9118-70c5a95410ef
      - Text Alignment Score: 78.95%

[176/212] Processing Question ID: test_fr_070
   -> Query Preview: "Peut-on reporter ses jours de vacances à l'année suivante ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 76598ba3-7524-4ae4-891c-68c3b95b7d3f
      - Text Alignment Score: 68.42%

[177/212] Processing Question ID: test_fr_071
   -> Query Preview: "Que se passe-t-il si un jour férié tombe pendant les vacances ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 7c5cc92c-8822-4270-9a95-3c8285421181
      - Text Alignment Score: 73.68%

[178/212] Processing Question ID: test_fr_072
   -> Query Preview: "Y a-t-il un congé prévu pour une assignation à témoigner au tribu..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: aa16bc97-a683-48c0-9bdd-c6eb3ce78828
      - Text Alignment Score: 76.47%

[179/212] Processing Question ID: test_fr_073
   -> Query Preview: "Qui détient les droits d'auteur sur les plans de cours ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: cd2ff341-dc83-4fcb-9aec-dbdc910bf703
      - Text Alignment Score: 52.94%

[180/212] Processing Question ID: test_fr_074
   -> Query Preview: "Qui possède les revenus d'une invention brevetable ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 754fe29c-e50d-4d85-bec1-6bfa2b99dfad
      - Text Alignment Score: 60.87%

[181/212] Processing Question ID: test_fr_075
   -> Query Preview: "Les professeurs peuvent-ils exercer des activités professionnelle..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 4a799824-23c4-436d-ac2e-66dd75624f6d
      - Text Alignment Score: 54.17%

[182/212] Processing Question ID: test_fr_076
   -> Query Preview: "Faut-il déclarer ses activités extérieures ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 4a799824-23c4-436d-ac2e-66dd75624f6d
      - Text Alignment Score: 64.00%

[183/212] Processing Question ID: test_fr_077
   -> Query Preview: "Quel est le rôle du directeur de département ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 339cd3ec-159f-4635-9e11-6fe16adf404f
      - Text Alignment Score: 57.14%

[184/212] Processing Question ID: test_fr_078
   -> Query Preview: "Comment devient-on directeur de département ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 2f45666c-3049-4410-bf80-fd628a513510
      - Text Alignment Score: 77.78%

[185/212] Processing Question ID: test_fr_079
   -> Query Preview: "Les directeurs reçoivent-ils une prime ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 5d944b18-f9f6-4e50-abbf-858ca1c48205
      - Text Alignment Score: 60.00%

[186/212] Processing Question ID: test_fr_080
   -> Query Preview: "Quel est l'âge de retraite obligatoire ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 84138ec8-394b-4179-8fb0-6c773e6fafaa
      - Text Alignment Score: 84.21%

[187/212] Processing Question ID: test_fr_081
   -> Query Preview: "La retraite progressive est-elle permise ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: e71212b8-39ef-4d13-9303-e909cf0184fe
      - Text Alignment Score: 57.89%

[188/212] Processing Question ID: test_fr_082
   -> Query Preview: "Qu'est-ce qu'une exigence financière (crise financière) ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 3171ace5-2bc0-4ce3-8598-8d474538e1bf
      - Text Alignment Score: 50.00%

[189/212] Processing Question ID: test_fr_083
   -> Query Preview: "Comment le contrat imposé a-t-il modifié les règles de l'exigence..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: c9b45c3c-682a-4925-8bb2-8a2fdcadd98b
      - Text Alignment Score: 52.38%

[190/212] Processing Question ID: test_fr_084
   -> Query Preview: "L'employeur est-il responsable de la santé et sécurité au travail..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 37871b57-a090-45f3-b873-c0266b357967
      - Text Alignment Score: 72.22%

[191/212] Processing Question ID: test_fr_085
   -> Query Preview: "Y a-t-il un comité de santé et sécurité ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 2615d75e-b836-4d13-a54f-c5bdc8a83945
      - Text Alignment Score: 62.50%

[192/212] Processing Question ID: test_fr_086
   -> Query Preview: "Les professeurs peuvent-ils être remplacés par des sous-traitants..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: de785e98-d802-4589-88cf-7e41dda67f60
      - Text Alignment Score: 63.16%

[193/212] Processing Question ID: test_fr_087
   -> Query Preview: "L'employeur paie-t-il les frais de déménagement des nouveaux enga..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 79e63ce6-1f1a-470f-a10d-9e778700ed39
      - Text Alignment Score: 77.78%

[194/212] Processing Question ID: test_fr_088
   -> Query Preview: "Qui détermine le curriculum des programmes d'études ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 2f45666c-3049-4410-bf80-fd628a513510
      - Text Alignment Score: 60.00%

[195/212] Processing Question ID: test_fr_089
   -> Query Preview: "Comment fonctionne une nomination conjointe (cross-appointment) ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 5f52aa99-13ae-47ea-86bb-81413164e4a7
      - Text Alignment Score: 89.47%

[196/212] Processing Question ID: test_fr_090
   -> Query Preview: "Un professeur a-t-il droit de vote au Sénat de l'université ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: cb560ba5-da96-4234-85ae-8d6ce5ebd94b
      - Text Alignment Score: 73.33%

[197/212] Processing Question ID: test_fr_091
   -> Query Preview: "Le Sénat de l'université participe-t-il aux négociations syndical..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 024e61d0-cd6a-46ae-bd2a-43cd8969e77a
      - Text Alignment Score: 55.56%

[198/212] Processing Question ID: test_fr_092
   -> Query Preview: "Que faire en cas de contradiction entre la convention et une poli..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: d41c6009-d050-45eb-9a93-00d30ae08463
      - Text Alignment Score: 75.00%

[199/212] Processing Question ID: test_fr_093
   -> Query Preview: "Quelles sont les obligations de l'Université Laurentienne face au..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a4b01a56-6835-4920-b8f6-a4136d392dde
      - Text Alignment Score: 65.00%

[200/212] Processing Question ID: test_fr_094
   -> Query Preview: "Les bibliothécaires professionnels sont-ils couverts par l'entent..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 381f8b88-241d-49ef-832f-af3ed5431f87
      - Text Alignment Score: 66.67%

[201/212] Processing Question ID: test_fr_095
   -> Query Preview: "Un bibliothécaire doit-il faire une période d'essai ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: e21239ca-13f8-4619-8d11-5eca9dfdfbd6
      - Text Alignment Score: 63.16%

[202/212] Processing Question ID: test_fr_096
   -> Query Preview: "Les professeurs à temps partiel ou chargés de cours font-ils part..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: b54121fe-b5f3-4cd7-ac9e-365074661e60
      - Text Alignment Score: 53.85%

[203/212] Processing Question ID: test_fr_097
   -> Query Preview: "Un membre peut-il démissionner sans préavis ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 33f472ff-4f47-4ef1-9a2f-c3494c0ec462
      - Text Alignment Score: 66.67%

[204/212] Processing Question ID: test_fr_098
   -> Query Preview: "L'employeur peut-il accéder aux données personnelles sur un ordin..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a070ebaf-d448-4911-b10e-ee671e89665b
      - Text Alignment Score: 60.00%

[205/212] Processing Question ID: test_fr_099
   -> Query Preview: "L'employeur est-il tenu de fournir de l'équipement de bureau ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: d121bb44-663e-451f-b195-78a8f38b3583
      - Text Alignment Score: 58.82%

[206/212] Processing Question ID: test_fr_100
   -> Query Preview: "La convention protège-t-elle les dénonciateurs (whistleblowers) ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: af86d12a-15d9-4690-97fa-a4568ed696dd
      - Text Alignment Score: 45.00%

[207/212] Processing Question ID: test_fr_101
   -> Query Preview: "Quelle est la durée d'application du contrat imposé ?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: a96793db-f78c-4dcb-846e-00814ff670e7
      - Text Alignment Score: 62.50%

[208/212] Processing Question ID: test_fr_102
   -> Query Preview: "Quelles sont les exigences concernant les heures de bureau pour l..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 49cec5f5-6c9a-41f6-b207-1b0226b8c5ae
      - Text Alignment Score: 76.92%

[209/212] Processing Question ID: test_fr_103
   -> Query Preview: "Quelle est la politique sur la liberté académique?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 6b38b311-44ec-4a16-848b-01cc89f893fd
      - Text Alignment Score: 87.50%

[210/212] Processing Question ID: test_fr_104
   -> Query Preview: "Comment la charge d'enseignement est-elle calculée?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 43d83be2-d3b7-401b-a811-3b3c760a4954
      - Text Alignment Score: 87.50%

[211/212] Processing Question ID: test_fr_105
   -> Query Preview: "Quelles sont les dispositions pour le congé sabbatique?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 15809a7f-ab91-4f76-917b-f897b8cfcf0d
      - Text Alignment Score: 61.54%

[212/212] Processing Question ID: test_fr_106
   -> Query Preview: "Quelle est la procédure de grief?..."
   -> Target Collection Table: Connect targeting -> 'multilingual_docs'
   -> SQL Context: No explicit article segment found. Filtering WHERE language == 'fr'
   ✅ Success: Chunk linked successfully.
      - Database Node UUID: 0c51f02c-b3de-4f93-a376-1182a37b4c3f
      - Text Alignment Score: 80.00%