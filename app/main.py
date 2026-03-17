from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from contextlib import asynccontextmanager
import shutil
import os
import uuid
import json
import asyncio
import tempfile

# Import our custom processors
from core.praat_processor import analyze_praat_features
from core.ast_processor import ASTPredictor
from core.quality_control import run_quality_checks
from core.audio_utils import get_speech_timestamps

# Initialize Model (Global Load on Startup)
MODEL_PATH = "models/regression_best_grbas_ast_model_fold_all.pt" 
ast_predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model
    global ast_predictor
    if os.path.exists(MODEL_PATH):
        ast_predictor = ASTPredictor(MODEL_PATH)
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}. AST features will fail.")
    yield
    # Unload the model
    ast_predictor = None

app = FastAPI(lifespan=lifespan)

# Serve the static HTML frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

@app.get("/adjectives")
def get_adjectives():
    with open("core/adjs_grouped.json") as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.get("/acoustic_norms")
def get_acoustic_norms():
    with open("core/acoustic_range.json") as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.post("/detect-speech")
async def detect_speech_endpoint(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    temp_filename_orig = os.path.join(temp_dir, f"temp_{uuid.uuid4()}_orig")
    temp_filename_wav = os.path.join(temp_dir, f"temp_{uuid.uuid4()}.wav")

    try:
        with open(temp_filename_orig, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ffmpeg_command = f"ffmpeg -i {temp_filename_orig} -y -hide_banner -loglevel error {temp_filename_wav}"
        process = await asyncio.create_subprocess_shell(
            ffmpeg_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        _, stderr = await process.communicate()

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {stderr.decode()}")
        
        timestamps = get_speech_timestamps(temp_filename_wav)
        return JSONResponse(content=timestamps)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/trim")
async def trim_audio_endpoint(file: UploadFile = File(...), start_sec: float = Form(...), end_sec: float = Form(...)):
    temp_dir = tempfile.mkdtemp()
    temp_id = uuid.uuid4()
    temp_filename_orig = os.path.join(temp_dir, f"temp_{temp_id}_orig")
    temp_filename_wav = os.path.join(temp_dir, f"temp_{temp_id}.wav")
    trimmed_filename = os.path.join(temp_dir, f"temp_{temp_id}_trimmed.wav")

    try:
        with open(temp_filename_orig, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ffmpeg_command = f"ffmpeg -i {temp_filename_orig} -y -hide_banner -loglevel error {temp_filename_wav}"
        process = await asyncio.create_subprocess_shell(
            ffmpeg_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {stderr.decode()}")

        trim_command = f"ffmpeg -i {temp_filename_wav} -ss {start_sec} -to {end_sec} -c copy {trimmed_filename}"
        trim_process = await asyncio.create_subprocess_shell(
            trim_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await trim_process.communicate()
        if trim_process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Trimming failed: {stderr.decode()}")

        return FileResponse(trimmed_filename, media_type="audio/wav", filename="trimmed.wav", background=BackgroundTask(lambda: shutil.rmtree(temp_dir, ignore_errors=True)))
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...), gender: str = Form(...), ignore_limit: bool = Form(False)):
    async def event_generator():
        temp_dir = tempfile.mkdtemp()
        temp_id = uuid.uuid4()
        temp_filename_orig = os.path.join(temp_dir, f"temp_{temp_id}_orig")
        analysis_file = os.path.join(temp_dir, f"temp_{temp_id}.wav")

        try:
            with open(temp_filename_orig, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            yield f"data: {json.dumps({'status': 'Processing audio file...'})}\n\n"
            await asyncio.sleep(0.01)
            
            ffmpeg_command = f"ffmpeg -i {temp_filename_orig} -y -hide_banner -loglevel error {analysis_file}"
            process = await asyncio.create_subprocess_shell(
                ffmpeg_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _, stderr = await process.communicate()

            if process.returncode != 0:
                error_message = stderr.decode()
                if "Invalid data found when processing input" in error_message:
                    error_to_show = "Could not decode audio. The file format may be unsupported."
                else:
                    error_to_show = f"FFmpeg conversion failed. Please ensure ffmpeg is installed. Error: {error_message}"
                yield f"event: error\ndata: {json.dumps({'error': error_to_show})}\n\n"
                return

            yield f"data: {json.dumps({'status': 'Starting quality checks...'})}\n\n"
            await asyncio.sleep(0.01)

            qc_passed = True
            # If ignore_limit is True, set max_duration to 1 hour (3600s)
            max_dur = 3600 if ignore_limit else 30
            for check_result in run_quality_checks(analysis_file, max_duration=max_dur):
                yield f"event: qc_step\ndata: {json.dumps(check_result)}\n\n"
                await asyncio.sleep(0.01)
                if check_result["status"] == "failed":
                    yield f"event: error\ndata: {json.dumps({'error': check_result['message']})}\n\n"
                    qc_passed = False
                    break
            
            if not qc_passed:
                yield "event: done\ndata: Analysis ended due to QC failure.\n\n"
                return

            yield f"data: {json.dumps({'status': 'Quality checks passed. Starting Praat analysis...'})}\n\n"
            await asyncio.sleep(0.01)

            try:
                praat_results = analyze_praat_features(analysis_file)
                yield f"data: {json.dumps({'status': 'Praat analysis complete. Starting AST analysis...'})}\n\n"
                await asyncio.sleep(0.01)
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': f'An error occurred during Praat analysis: {e}'})}\n\n"
                yield "event: done\ndata: Analysis failed.\n\n"
                return

            ast_results = {}
            if ast_predictor:
                try:
                    ast_results = ast_predictor.predict(analysis_file)
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'error': f'An error occurred during AST analysis: {e}'})}\n\n"
                    yield "event: done\ndata: Analysis failed.\n\n"
                    return
            else:
                ast_results = {"error": "Model not loaded"}

            yield f"data: {json.dumps({'status': 'AST analysis complete. Finishing up...'})}\n\n"
            await asyncio.sleep(0.01)
            
            final_results = {
                "filename": file.filename,
                "gender": gender,
                "acoustic_features": praat_results,
                "semantic_features": ast_results
            }
            yield f"event: result\ndata: {json.dumps(final_results)}\n\n"
            yield "event: done\ndata: Analysis complete.\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': f'An unexpected error occurred: {e}'})}\n\n"
            yield "event: done\ndata: Analysis failed.\n\n"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return StreamingResponse(event_generator(), media_type="text/event-stream")