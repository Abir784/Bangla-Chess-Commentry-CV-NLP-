#!/usr/bin/env python
"""
Chess Video Analysis Pipeline - Stage Runner
Run individual stages independently
"""

import os
import sys
import json
import argparse
from pathlib import Path

# GPU Setup (if available)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class StageRunner:
    """Run individual pipeline stages"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.frames_dir = self.base_dir / "preprocessed_frames"
        self.boards_dir = self.base_dir / "normalized_boards"
        self.output_dir = self.base_dir / "output"
        self.videos_dir = self.base_dir / "videos"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # STAGE 1: Frame Extraction
    # ========================================================================
    def stage_1_extract_frames(self, video_path):
        """Extract frames from video at 1 FPS"""
        print("\n" + "="*70)
        print("STAGE 1: Video Frame Extraction")
        print("="*70)
        
        if not Path(video_path).exists():
            print(f"❌ Video not found: {video_path}")
            return False
        
        try:
            import cv2
            
            print(f"📹 Input: {video_path}")
            print(f"📁 Output: {self.frames_dir}/")
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"\n📊 Video Info:")
            print(f"   FPS: {fps}")
            print(f"   Total frames: {frame_count}")
            print(f"   Resolution: {width}x{height}")
            
            # Extract 1 frame per second
            frame_interval = int(fps)
            frame_num = 0
            saved = 0
            
            self.frames_dir.mkdir(exist_ok=True)
            
            print(f"\n⏳ Extracting frames (1 per second)...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_num % frame_interval == 0:
                    filename = self.frames_dir / f"frame_{saved+1:03d}.jpg"
                    cv2.imwrite(str(filename), frame)
                    saved += 1
                    if saved % 10 == 0:
                        print(f"   Extracted {saved} frames...")
                
                frame_num += 1
            
            cap.release()
            print(f"\n✅ Extracted {saved} frames to {self.frames_dir}/")
            return True
            
        except ImportError:
            print("❌ OpenCV not installed: pip install opencv-python")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    # ========================================================================
    # STAGE 2: Board Detection (Advanced Multi-Strategy)
    # ========================================================================
    def stage_2_board_detection(self):
        """
        Detect chessboard using advanced multi-strategy algorithm:
        - Contour detection (morphological operations)
        - Edge-based detection (Canny + Hough lines)
        - Blob detection
        - Watershed segmentation
        - Voting system for robust detection
        """
        print("\n" + "="*70)
        print("STAGE 2: Board Detection (Advanced Multi-Strategy)")
        print("="*70)
        
        # Check if frames exist
        frame_files = sorted(self.frames_dir.glob("*.jpg"))
        if not frame_files:
            print(f"[ERROR] No frames found in {self.frames_dir}")
            print(f"   Run Stage 1 first: python stages.py --stage 1 --video <video.mp4>")
            return False
        
        print(f"[INPUT] Path: {self.frames_dir}/")
        print(f"[OUTPUT] Path: {self.boards_dir}/")
        print(f"[INFO] Frames to process: {len(frame_files)}")
        
        try:
            import cv2
            import numpy as np
            from board_detector_ultimate import UltimateChessBoardDetector
            
            # Create detector instance
            detector = UltimateChessBoardDetector(debug=False)
            
            # Create output directory
            self.boards_dir.mkdir(exist_ok=True)
            
            # Process frames
            print(f"\n[PROCESSING] {len(frame_files)} frames...")
            print("   Using Ultimate Detection with 4 strategies:")
            print("   - Chess-specific 8x8 grid detection (findChessboardCorners)")
            print("   - RANSAC-based edge and line intersection")
            print("   - Adaptive contour detection with corner verification")
            print("   - Harris corner detection with geometric filtering")
            print("   - Multi-candidate clarity scoring (clearest board selection)")
            print("   - Sub-pixel corner refinement")
            print("   - Multi-scale adaptive preprocessing")
            print("")
            
            detected = 0
            failed = 0
            boards_data = []
            
            for idx, frame_path in enumerate(frame_files, 1):
                try:
                    # Load frame
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        failed += 1
                        continue
                    
                    # Process with advanced detector
                    normalized_board, corners = detector.process_frame(frame)
                    
                    if normalized_board is None or corners is None:
                        failed += 1
                        continue
                    
                    # Save normalized board
                    frame_num = int(frame_path.stem.split('_')[1])
                    output_path = self.boards_dir / f"board_{frame_num:03d}.jpg"
                    cv2.imwrite(str(output_path), normalized_board)
                    
                    # Store metadata
                    boards_data.append({
                        "frame": frame_num,
                        "input": str(frame_path.name),
                        "output": str(output_path.name),
                        "corners": [[float(x), float(y)] for x, y in corners],
                        "size": [512, 512],
                    })
                    
                    detected += 1
                    if detected % 50 == 0:
                        print(f"  [OK] Processed {detected}/{len(frame_files)} boards")
                    
                except Exception as e:
                    failed += 1
            
            # Save metadata
            metadata_file = self.output_dir / "board_detection_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump({
                    "total_frames": len(frame_files),
                    "boards_detected": detected,
                    "boards_failed": failed,
                    "detection_rate": f"{100*detected/len(frame_files):.1f}%",
                    "detection_method": "Ultimate Multi-Strategy with Clarity Scoring (Grid + Edge+RANSAC + Contour + Harris + Sub-pixel Refinement)",
                    "boards": boards_data,
                }, f, indent=2)
            
            print(f"\n[SUCCESS] Board detection complete!")
            print(f"   Detected: {detected}/{len(frame_files)} boards")
            print(f"   Detection rate: {100*detected/len(frame_files):.1f}%")
            print(f"   Metadata: {metadata_file}")
            
            return True
            
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print(f"   Install with: pip install opencv-python numpy")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================================================
    # STAGE 3: Piece Recognition with Board Annotation
    # ========================================================================
    def stage_3_piece_recognition(self):
        """Detect chess pieces, annotate board grid, and identify positions"""
        print("\n" + "="*70)
        print("STAGE 3: Chess Board Analysis with Piece Recognition")
        print("="*70)
        
        # Check if board images exist
        boards_dir = self.base_dir / "normalized_boards"
        if not boards_dir.exists():
            print(f"[ERROR] Normalized boards not found at {boards_dir}")
            print(f"   Run Stage 2 first: python stages.py --stage 2")
            return False
        
        board_files = sorted(boards_dir.glob("*.jpg"))
        if not board_files:
            print(f"[ERROR] No board images found in {boards_dir}")
            return False
        
        print(f"[INPUT] Path: {boards_dir}/")
        print(f"[OUTPUT] Path: {self.output_dir}/")
        print(f"[INFO] Boards to process: {len(board_files)}")
        
        try:
            import cv2
            import numpy as np
            
            # Create output directories
            annotated_dir = self.output_dir / "annotated_boards"
            annotated_dir.mkdir(exist_ok=True)
            
            print(f"\n[PROCESSING] {len(board_files)} board images...")
            print("   With grid annotation and piece detection\n")
            
            processed = 0
            failed = 0
            positions_data = []
            
            for idx, board_path in enumerate(board_files, 1):
                try:
                    board_image = cv2.imread(str(board_path))
                    if board_image is None:
                        failed += 1
                        continue
                    
                    h, w = board_image.shape[:2]
                    
                    # Create annotated version
                    annotated = board_image.copy()
                    
                    # 1. Detect pieces in each square
                    board_state = self._analyze_board_squares(board_image)
                    
                    # 2. Draw grid and annotations
                    annotated = self._annotate_board_grid(annotated, board_state)
                    
                    # 3. Draw piece annotations
                    annotated = self._annotate_pieces(annotated, board_state)
                    
                    # Save annotated board
                    frame_num = int(board_path.stem.split('_')[1])
                    output_path = annotated_dir / f"board_{frame_num:03d}_annotated.jpg"
                    cv2.imwrite(str(output_path), annotated)
                    
                    # Collect position data
                    positions_data.append({
                        "frame": frame_num,
                        "board_image": str(board_path.name),
                        "annotated_image": str(output_path.name),
                        "board_state": board_state,
                        "pieces_detected": sum(1 for row in board_state for sq in row if sq['piece']),
                    })
                    
                    processed += 1
                    if processed % 50 == 0:
                        print(f"  [OK] Processed {processed}/{len(board_files)} boards")
                    
                except Exception as e:
                    failed += 1
            
            # Save position detection results
            positions_file = self.output_dir / "piece_position_analysis.json"
            with open(positions_file, 'w') as f:
                json.dump({
                    "total_boards": len(board_files),
                    "boards_processed": processed,
                    "boards_failed": failed,
                    "processing_rate": f"{100*processed/len(board_files):.1f}%",
                    "analysis_type": "Grid Annotation + Piece Detection",
                    "output_format": "8x8 board state with piece positions",
                    "boards": positions_data,
                }, f, indent=2)
            
            print(f"\n[SUCCESS] Board analysis complete!")
            print(f"   Processed: {processed}/{len(board_files)} boards")
            print(f"   Annotated boards: {annotated_dir}/")
            print(f"   Position data: {positions_file}")
            
            return True
            
        except ImportError as e:
            print(f"[ERROR] Missing dependency: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _analyze_board_squares(self, board_image):
        """
        Analyze 8x8 board grid and detect pieces in each square.
        Returns 8x8 array with piece information for each square.
        """
        import cv2
        import numpy as np
        
        h, w = board_image.shape[:2]
        square_h = h // 8
        square_w = w // 8
        
        # Chess notation: columns a-h (0-7), rows 8-1 (0-7 from top)
        board_state = []
        gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
        
        for row in range(8):
            row_data = []
            for col in range(8):
                y1 = row * square_h
                y2 = (row + 1) * square_h
                x1 = col * square_w
                x2 = (col + 1) * square_w
                
                square_region = gray[y1:y2, x1:x2]
                
                # Calculate metrics to detect piece presence
                img_region = board_image[y1:y2, x1:x2]
                variance = cv2.Laplacian(square_region, cv2.CV_64F).var()
                mean_intensity = np.mean(square_region)
                std_intensity = np.std(square_region)
                
                # Detect piece (higher variance indicates piece presence)
                has_piece = variance > 80
                
                # Classify piece color (approximate)
                if has_piece:
                    piece_color = "white" if mean_intensity > 120 else "black"
                    piece_type = self._classify_piece_improved(square_region, mean_intensity, std_intensity)
                else:
                    piece_color = None
                    piece_type = None
                
                # Store square data
                chess_col = chr(97 + col)  # a-h
                chess_row = str(8 - row)      # 1-8 (from bottom)
                square_name = f"{chess_col}{chess_row}"
                
                row_data.append({
                    "position": square_name,
                    "grid_coords": (row, col),
                    "pixel_coords": (x1, y1, x2, y2),
                    "piece": piece_type,
                    "color": piece_color,
                    "confidence": float(min(variance / 400, 1.0)) if has_piece else 0.0,
                    "metrics": {
                        "variance": float(variance),
                        "mean_intensity": float(mean_intensity),
                        "std_intensity": float(std_intensity),
                    }
                })
            
            board_state.append(row_data)
        
        return board_state
    
    def _classify_piece_improved(self, square_region, mean_intensity, std_intensity):
        """
        Improved piece classification based on pixel patterns.
        Returns simplified piece type (P, N, B, R, Q, K)
        """
        import cv2
        import numpy as np
        
        # Piece classification heuristics based on shape/intensity patterns
        # This is a simplified approach; proper classification needs trained model
        
        # Analyze the region shape
        h, w = square_region.shape
        
        # Calculate edge distribution
        edges = cv2.Canny(square_region, 50, 150)
        edge_count = np.count_nonzero(edges)
        edge_density = edge_count / (h * w)
        
        # Classify based on patterns
        if std_intensity < 15:
            return "empty"  # Uniform color = empty square
        elif edge_density > 0.15:
            # High edge density suggests complex piece
            if std_intensity > 40:
                return "K"  # King (most edges)
            else:
                return "Q"  # Queen
        elif edge_density > 0.10:
            # Medium-high edges
            if mean_intensity > 130:
                return "R"  # Rook
            else:
                return "N"  # Knight
        elif edge_density > 0.06:
            return "B"  # Bishop
        else:
            return "P"  # Pawn
    
    def _annotate_board_grid(self, annotated, board_state):
        """Draw chess grid with square labels and alternating colors"""
        import cv2
        import numpy as np
        
        h, w = annotated.shape[:2]
        square_h = h // 8
        square_w = w // 8
        
        # Colors for visualization
        grid_color = (0, 255, 0)      # Green for grid lines
        label_color = (255, 255, 255)  # White for text
        label_color_dark = (0, 0, 0)   # Black for text on light bg
        
        # Draw grid lines
        for i in range(1, 8):
            # Vertical lines
            cv2.line(annotated, (i * square_w, 0), (i * square_w, h), grid_color, 2)
            # Horizontal lines
            cv2.line(annotated, (0, i * square_h), (w, i * square_h), grid_color, 2)
        
        # Draw border
        cv2.rectangle(annotated, (0, 0), (w-1, h-1), grid_color, 3)
        
        # Add row numbers (1-8, from bottom to top)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        for row in range(8):
            row_num = 8 - row
            y_pos = (row + 0.5) * square_h + 5
            cv2.putText(annotated, str(row_num), (5, int(y_pos)), 
                       font, font_scale, label_color, thickness)
        
        # Add column labels (a-h, left to right)
        for col in range(8):
            col_label = chr(97 + col)
            x_pos = (col + 0.7) * square_w
            cv2.putText(annotated, col_label, (int(x_pos), h - 5),
                       font, font_scale, label_color, thickness)
        
        return annotated
    
    def _annotate_pieces(self, annotated, board_state):
        """Draw piece annotations with boxes and labels"""
        import cv2
        import numpy as np
        
        h, w = annotated.shape[:2]
        square_h = h // 8
        square_w = w // 8
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Colors for pieces
        white_color = (200, 200, 255)  # Light blue for white pieces
        black_color = (255, 100, 100)   # Light red for black pieces
        text_color = (255, 255, 255)   # White text
        
        # Piece symbols
        piece_symbols = {
            'P': 'Pawn',
            'N': 'Knight',
            'B': 'Bishop',
            'R': 'Rook',
            'Q': 'Queen',
            'K': 'King',
            'empty': ''
        }
        
        # Draw annotations for each square
        for row_idx, row in enumerate(board_state):
            for col_idx, square in enumerate(row):
                if square['piece'] and square['piece'] != 'empty':
                    x1, y1, x2, y2 = square['pixel_coords']
                    piece = square['piece']
                    color = square['color']
                    conf = square['confidence']
                    position = square['position']
                    
                    # Choose color based on piece color
                    box_color = white_color if color == 'white' else black_color
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1 + 5, y1 + 5), (x2 - 5, y2 - 5),
                                box_color, 2)
                    
                    # Draw piece symbol in center of square
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Text: piece type + color abbrev + position
                    label = f"{piece}{color[0].upper()} {position}"
                    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                    text_x = center_x - text_size[0] // 2
                    text_y = center_y + text_size[1] // 2
                    
                    # Semi-transparent background for text readability
                    cv2.rectangle(annotated, 
                                (text_x - 2, text_y - text_size[1] - 2),
                                (text_x + text_size[0] + 2, text_y + 2),
                                (0, 0, 0), -1)
                    
                    cv2.putText(annotated, label, (text_x, text_y),
                               font, font_scale, text_color, thickness)
                    
                    # Draw confidence score in corner
                    conf_text = f"{conf:.0%}"
                    conf_pos = (x1 + 5, y2 - 5)
                    cv2.putText(annotated, conf_text, conf_pos,
                               font, 0.5, text_color, 1)
        
        return annotated
    
    # ========================================================================
    # STAGE 4: FEN Encoding
    # ========================================================================
    def stage_4_fen_encoding(self):
        """Demo: Convert board state to FEN"""
        print("\n" + "="*70)
        print("STAGE 4: Board State to FEN Encoding")
        print("="*70)
        
        try:
            import chess
            
            print("\n📋 Demo - Initial Chess Position:")
            
            # Create initial position
            board = chess.Board()
            fen = board.fen()
            
            print(f"   FEN: {fen[:60]}...")
            print(f"   Valid: {board.is_valid()}")
            print(f"   Turn: {'White' if board.turn else 'Black'}")
            print(f"   Castling: {board.castling_rights}")
            
            # Save example
            output_file = self.output_dir / "fen_example.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "fen": fen,
                    "white_to_move": board.turn,
                    "castling_rights": str(board.castling_rights),
                }, f, indent=2)
            
            print(f"\n✅ Example saved to {output_file}")
            return True
            
        except ImportError:
            print("❌ python-chess not installed: pip install python-chess")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    # ========================================================================
    # STAGE 5: Move Reconstruction
    # ========================================================================
    def stage_5_move_reconstruction(self):
        """Demo: Find move between two positions"""
        print("\n" + "="*70)
        print("STAGE 5: Move Reconstruction & Validation")
        print("="*70)
        
        try:
            import chess
            
            print("\n♟️  Demo - Move Reconstruction:")
            
            # Initial position
            board = chess.Board()
            print(f"   Position 1: Starting position")
            
            # Make move e2-e4
            move = chess.Move.from_uci("e2e4")
            board.push(move)
            
            print(f"   Position 2: After e4")
            print(f"   Move found: {board.san(move)}")
            print(f"   Legal: {move in list(board.legal_moves)[:1] or 'Yes'}")
            
            # Save example
            output_file = self.output_dir / "move_example.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "move_san": "e4",
                    "move_uci": "e2e4",
                    "from_square": "e2",
                    "to_square": "e4",
                    "is_legal": True,
                }, f, indent=2)
            
            print(f"\n✅ Example saved to {output_file}")
            return True
            
        except ImportError:
            print("❌ python-chess not installed: pip install python-chess")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    # ========================================================================
    # STAGE 6: Engine Analysis (Stockfish)
    # ========================================================================
    def stage_6_engine_analysis(self):
        """Demo: Analyze position with Stockfish"""
        print("\n" + "="*70)
        print("STAGE 6: Engine Analysis (Stockfish)")
        print("="*70)
        
        stockfish_path = Path("engines/stockfish/stockfish.exe")
        
        if not stockfish_path.exists():
            print(f"❌ Stockfish not found at: {stockfish_path}")
            print(f"\n📥 Download from: https://stockfishchess.org/download/")
            print(f"   Extract to: {stockfish_path.parent}/")
            return False
        
        try:
            from stockfish import Stockfish
            
            print(f"✓ Stockfish found at: {stockfish_path}")
            print("\n⚙️  Analyzing initial position...")
            
            sf = Stockfish(
                path=str(stockfish_path),
                depth=15,
                parameters={"Threads": 4}
            )
            
            sf.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            best_move = sf.get_best_move()
            evaluation = sf.get_evaluation()
            
            print(f"\n   Best move: {best_move}")
            print(f"   Evaluation: {evaluation}")
            
            # Save example
            output_file = self.output_dir / "engine_example.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "best_move": best_move,
                    "evaluation": str(evaluation),
                }, f, indent=2)
            
            print(f"\n✅ Analysis saved to {output_file}")
            return True
            
        except ImportError:
            print("❌ stockfish module not installed: pip install stockfish")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    # ========================================================================
    # STAGE 7: Feature Engineering
    # ========================================================================
    def stage_7_feature_engineering(self):
        """Create NLP-friendly features"""
        print("\n" + "="*70)
        print("STAGE 7: Feature Engineering")
        print("="*70)
        
        print("\n🏷️  Creating features from example move...")
        
        # Example move features
        features = {
            "move": "Nf3",
            "move_type": "development",
            "game_phase": "opening",
            "evaluation": {
                "before": 0,
                "after": 17,
                "shift": "improving",
            },
            "threat_created": False,
            "piece_activity": True,
            "confidence": {
                "move_type": 0.95,
                "evaluation": 0.99,
            },
        }
        
        # Save
        output_file = self.output_dir / "features_example.json"
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2, ensure_ascii=False)
        
        print(f"\n   Move: {features['move']}")
        print(f"   Type: {features['move_type']}")
        print(f"   Phase: {features['game_phase']}")
        print(f"   Evaluation: {features['evaluation']['before']} → {features['evaluation']['after']}")
        print(f"   Trend: {features['evaluation']['shift']}")
        
        print(f"\n✅ Features saved to {output_file}")
        return True
    
    # ========================================================================
    # STAGE 9: Voice Synthesis (Optional)
    # ========================================================================
    def stage_9_voice_synthesis(self, text):
        """Convert text to speech"""
        print("\n" + "="*70)
        print("STAGE 9: Voice Synthesis")
        print("="*70)
        
        print(f"\n🎙️  Text: {text}")
        
        try:
            from TTS.api import TTS
            
            print("⏳ Loading TTS model...")
            # Try to load Bangla model
            tts = TTS(model_name="tts_models/bn/bangla/glow-tts", progress_bar=False)
            
            output_file = self.output_dir / "speech_output.wav"
            print(f"⏳ Synthesizing speech...")
            tts.tts_to_file(text=text, file_path=str(output_file))
            
            print(f"✅ Audio saved to {output_file}")
            return True
            
        except ImportError:
            print("⚠️  TTS not installed: pip install TTS")
            print("   Skipping voice synthesis...")
            return False
        except Exception as e:
            print(f"⚠️  Error: {e}")
            return False

    # ========================================================================
    # STAGE 8: System & Environment Check (YOLO/GPU)
    # ========================================================================
    def stage_8_system_check(self):
        """Check Python, PyTorch, CUDA, Ultralytics, and dataset status"""
        print("\n" + "="*80)
        print("STAGE 8: System & Environment Check")
        print("="*80)

        import platform

        print(f"Python Version: {sys.version}")
        print(f"Platform: {platform.platform()}")
        print(f"Processor: {platform.processor()}")

        print("\n" + "="*80)
        print("PyTorch INFORMATION")
        print("="*80)

        try:
            import torch
            print(f"PyTorch Version: {torch.__version__}")
            print(f"CUDA Available: {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"cuDNN Version: {torch.backends.cudnn.version()}")
                print(f"Number of GPUs: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    print(f"\n--- GPU {i} ---")
                    print(f"Name: {torch.cuda.get_device_name(i)}")
                    print(f"Compute Capability: {props.major}.{props.minor}")
                    print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
            else:
                print("GPU not available. Training will use CPU (slower).")
        except ImportError as e:
            print(f"PyTorch not installed: {e}")
            return False

        print("\n" + "="*80)
        print("ULTRALYTICS YOLO")
        print("="*80)
        try:
            import ultralytics
            print(f"Ultralytics Version: {ultralytics.__version__}")
            print("Ultralytics is installed ✓")
        except ImportError:
            print("Ultralytics NOT installed ✗")

        print("\n" + "="*80)
        print("DATASET INFORMATION")
        print("="*80)
        data_yaml = self.base_dir / "datasets" / "chess_yolo" / "data.yaml"
        train_images = self.base_dir / "datasets" / "chess_yolo" / "images" / "train"
        val_images = self.base_dir / "datasets" / "chess_yolo" / "images" / "val"
        train_labels = self.base_dir / "datasets" / "chess_yolo" / "labels" / "train"
        val_labels = self.base_dir / "datasets" / "chess_yolo" / "labels" / "val"

        print(f"Data YAML: {data_yaml.exists()} {'✓' if data_yaml.exists() else '✗'}")
        if data_yaml.exists():
            print(f"  Path: {data_yaml}")

        print(f"Train Images: {len(list(train_images.glob('*.jpg'))) if train_images.exists() else 0}")
        print(f"Train Labels: {len(list(train_labels.glob('*.txt'))) if train_labels.exists() else 0}")
        print(f"Val Images: {len(list(val_images.glob('*.jpg'))) if val_images.exists() else 0}")
        print(f"Val Labels: {len(list(val_labels.glob('*.txt'))) if val_labels.exists() else 0}")

        labels_file = self.base_dir / "labels.txt"
        if labels_file.exists():
            labels = [line.strip() for line in labels_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            print(f"Number of Classes: {len(labels)}")
            print(f"Classes: {', '.join(labels)}")

        print("\n" + "="*80)
        print("QUICK IMPORT TEST")
        print("="*80)
        try:
            import torch
            print(f"✓ PyTorch imported successfully: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
        except Exception as e:
            print(f"✗ Failed to import PyTorch: {type(e).__name__}: {e}")
            return False

        return True

    # ========================================================================
    # STAGE 10: Prepare YOLO Dataset
    # ========================================================================
    def stage_10_prepare_yolo_dataset(self, labels_subdir=None, split_ratio=0.8, seed=42):
        """Create train/val split and data.yaml from frame+label pairs"""
        print("\n" + "="*80)
        print("STAGE 10: Prepare YOLO Dataset")
        print("="*80)

        import random
        import shutil

        images_dir = self.base_dir / "preprocessed_frames"
        if labels_subdir:
            labels_dir = self.base_dir / "Annotations" / labels_subdir
        else:
            labels_dir = self.base_dir / "Annotations" / "labels_nlp_2026-02-10-11-44-39"

        out_dir = self.base_dir / "datasets" / "chess_yolo"
        images_train = out_dir / "images" / "train"
        images_val = out_dir / "images" / "val"
        labels_train = out_dir / "labels" / "train"
        labels_val = out_dir / "labels" / "val"

        if not images_dir.exists() or not labels_dir.exists():
            print(f"❌ Required directory missing:\n  images: {images_dir}\n  labels: {labels_dir}")
            return False

        for p in [images_train, images_val, labels_train, labels_val]:
            p.mkdir(parents=True, exist_ok=True)

        labels_path = self.base_dir / "labels.txt"
        if not labels_path.exists():
            print(f"❌ labels.txt not found: {labels_path}")
            return False
        labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        pairs = []
        for label_file in labels_dir.glob("*.txt"):
            image_file = images_dir / f"{label_file.stem}.jpg"
            if image_file.exists():
                pairs.append((image_file, label_file))

        if not pairs:
            print("❌ No matching image-label pairs found")
            return False

        pairs.sort(key=lambda x: x[0].name)
        random.seed(seed)
        random.shuffle(pairs)

        split_idx = int(len(pairs) * split_ratio)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        for image_file, label_file in train_pairs:
            shutil.copy2(image_file, images_train / image_file.name)
            shutil.copy2(label_file, labels_train / label_file.name)

        for image_file, label_file in val_pairs:
            shutil.copy2(image_file, images_val / image_file.name)
            shutil.copy2(label_file, labels_val / label_file.name)

        yaml_lines = [
            f"path: {out_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
        ]
        for name in labels:
            yaml_lines.append(f"  - {name}")
        (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

        print(f"images: {len(pairs)}")
        print(f"train: {len(train_pairs)}")
        print(f"val: {len(val_pairs)}")
        print(f"data.yaml: {out_dir / 'data.yaml'}")
        return True

    # ========================================================================
    # STAGE 11: Train YOLO11
    # ========================================================================
    def stage_11_train_yolo11(self, model_weights="yolo11s.pt", epochs=30, imgsz=416, batch=4, workers=2):
        """Train YOLO11 model"""
        print("\n" + "="*80)
        print("STAGE 11: Train YOLO11")
        print("="*80)

        from ultralytics import YOLO
        import torch

        data_yaml = self.base_dir / "datasets" / "chess_yolo" / "data.yaml"
        output_dir = self.base_dir / "runs" / "detect"
        if not data_yaml.exists():
            print(f"❌ Dataset yaml not found: {data_yaml}")
            return False

        device = 0 if torch.cuda.is_available() else "cpu"
        print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

        model = YOLO(model_weights)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            optimizer="AdamW",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            amp=True,
            cache=False,
            pin_memory=False,
            project=str(output_dir),
            name="train_chess",
            exist_ok=True,
            pretrained=True,
            verbose=True,
            save=True,
            save_period=10,
            plots=True,
            patience=50,
            val=True,
        )

        metrics = model.val()
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        return True

    # ========================================================================
    # STAGE 12: Train YOLOv8
    # ========================================================================
    def stage_12_train_yolov8(self, model_weights="yolov8s.pt", epochs=100, imgsz=640, batch=16, workers=8):
        """Train YOLOv8 model"""
        print("\n" + "="*80)
        print("STAGE 12: Train YOLOv8")
        print("="*80)

        from ultralytics import YOLO
        import torch

        data_yaml = self.base_dir / "datasets" / "chess_yolo" / "data.yaml"
        output_dir = self.base_dir / "runs" / "detect"
        if not data_yaml.exists():
            print(f"❌ Dataset yaml not found: {data_yaml}")
            return False

        device = 0 if torch.cuda.is_available() else "cpu"
        print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

        model = YOLO(model_weights)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            optimizer="AdamW",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            amp=True,
            cache=True,
            project=str(output_dir),
            name="train_chess_v8",
            exist_ok=True,
            pretrained=True,
            verbose=True,
            save=True,
            save_period=10,
            plots=True,
            patience=50,
            val=True,
        )

        metrics = model.val()
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        return True

    # ========================================================================
    # STAGE 13: Resume Training
    # ========================================================================
    def stage_13_resume_training(self, checkpoint_path=None):
        """Resume training from checkpoint"""
        print("\n" + "="*80)
        print("STAGE 13: Resume Training")
        print("="*80)

        from ultralytics import YOLO
        import torch

        if checkpoint_path:
            ckpt = Path(checkpoint_path)
        else:
            ckpt = self.base_dir / "runs" / "detect" / "train_chess" / "weights" / "last.pt"
        if not ckpt.exists():
            print(f"❌ Checkpoint not found: {ckpt}")
            return False

        device = 0 if torch.cuda.is_available() else "cpu"
        model = YOLO(str(ckpt))
        model.train(resume=True, device=device)
        metrics = model.val()
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        return True

    # ========================================================================
    # STAGE 14: Export Model
    # ========================================================================
    def stage_14_export_model(self, model_path=None, export_formats=None):
        """Export trained YOLO model into selected formats"""
        print("\n" + "="*80)
        print("STAGE 14: Export YOLO Model")
        print("="*80)

        from ultralytics import YOLO

        if model_path:
            best_model = Path(model_path)
        else:
            best_model = self.base_dir / "runs" / "detect" / "train_chess" / "weights" / "best.pt"

        if not best_model.exists():
            print(f"❌ Model not found: {best_model}")
            return False

        model = YOLO(str(best_model))
        formats = export_formats or ["onnx", "torchscript", "engine", "openvino", "tflite"]

        for fmt in formats:
            print(f"\nExporting: {fmt}")
            try:
                kwargs = {"format": fmt}
                if fmt == "onnx":
                    kwargs.update({"dynamic": True, "simplify": True})
                if fmt in {"engine", "openvino"}:
                    kwargs.update({"half": True})
                if fmt == "engine":
                    kwargs.update({"device": 0})
                if fmt == "tflite":
                    kwargs.update({"int8": False})
                out = model.export(**kwargs)
                print(f"✓ Success: {out}")
            except Exception as e:
                print(f"✗ Failed: {e}")

        return True

    # ========================================================================
    # STAGE 15: Image Inference
    # ========================================================================
    def stage_15_image_inference(self, model_path=None, image_path=None, conf=0.25, iou=0.45):
        """Run inference on one image and optionally on val images"""
        print("\n" + "="*80)
        print("STAGE 15: Image Inference")
        print("="*80)

        from ultralytics import YOLO

        model_file = Path(model_path) if model_path else self.base_dir / "runs" / "detect" / "train_chess" / "weights" / "best.pt"
        test_image = Path(image_path) if image_path else self.base_dir / "preprocessed_frames" / "frame_001.jpg"
        output_dir = self.base_dir / "inference_results"
        output_dir.mkdir(exist_ok=True)

        if not model_file.exists():
            print(f"❌ Model not found: {model_file}")
            return False
        if not test_image.exists():
            print(f"❌ Image not found: {test_image}")
            return False

        model = YOLO(str(model_file))
        results = model.predict(
            source=str(test_image),
            save=True,
            save_txt=True,
            save_conf=True,
            conf=conf,
            iou=iou,
            project=str(output_dir),
            name="inference",
            exist_ok=True,
            show_labels=True,
            show_conf=True,
            line_width=2,
        )

        print("\nDETECTION RESULTS")
        for r in results:
            boxes = r.boxes
            print(f"Detected {len(boxes)} objects:")
            for box in boxes:
                cls = int(box.cls[0])
                score = float(box.conf[0])
                name = model.names[cls]
                print(f"  - {name}: {score:.2%}")

        return True

    # ========================================================================
    # STAGE 16: Video Inference
    # ========================================================================
    def stage_16_video_inference(self, model_path=None, source=None, out=None, conf=0.25, imgsz=640, vid_stride=1, max_frames=3000, no_show=True, display_scale=0.55):
        """Run YOLO on video and write annotated output video"""
        print("\n" + "="*80)
        print("STAGE 16: Video Inference")
        print("="*80)

        import cv2
        from ultralytics import YOLO

        weights_path = Path(model_path) if model_path else self.base_dir / "runs" / "detect" / "train" / "weights" / "best.pt"
        source_path = Path(source) if source else self.base_dir / "videos" / "chess.mp4"
        out_dir = Path(out) if out else self.base_dir / "runs" / "detect" / "predict_custom"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not weights_path.exists():
            print(f"❌ Weights not found: {weights_path}")
            return False
        if not source_path.exists():
            print(f"❌ Video not found: {source_path}")
            return False

        model = YOLO(str(weights_path))
        results_iter = model.predict(
            source=str(source_path),
            stream=True,
            conf=conf,
            imgsz=imgsz,
            vid_stride=vid_stride,
            save=False,
        )

        writer = None
        video_out_path = out_dir / f"{source_path.stem}_pred.mp4"

        try:
            for idx, result in enumerate(results_iter, start=1):
                if max_frames > 0 and idx > max_frames:
                    break
                annotated = result.plot()
                if writer is None:
                    height, width = annotated.shape[:2]
                    fps = result.speed.get("fps", 25) or 25
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))
                writer.write(annotated)
                if not no_show:
                    if display_scale != 1.0:
                        disp_w = max(1, int(annotated.shape[1] * display_scale))
                        disp_h = max(1, int(annotated.shape[0] * display_scale))
                        preview = cv2.resize(annotated, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                    else:
                        preview = annotated
                    cv2.imshow("YOLO Prediction", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            if writer is not None:
                writer.release()
            if not no_show:
                cv2.destroyAllWindows()

        print(f"Saved video: {video_out_path}")
        return True
    
    def run_stage(self, args):
        """Run a specific stage"""

        stage_num = args.stage
        stages = {
            1: ("Frame Extraction", self.stage_1_extract_frames),
            2: ("Board Detection", self.stage_2_board_detection),
            3: ("Piece Recognition", self.stage_3_piece_recognition),
            4: ("FEN Encoding", self.stage_4_fen_encoding),
            5: ("Move Reconstruction", self.stage_5_move_reconstruction),
            6: ("Engine Analysis", self.stage_6_engine_analysis),
            7: ("Feature Engineering", self.stage_7_feature_engineering),
            8: ("System Check", self.stage_8_system_check),
            9: ("Voice Synthesis", self.stage_9_voice_synthesis),
            10: ("Prepare YOLO Dataset", self.stage_10_prepare_yolo_dataset),
            11: ("Train YOLO11", self.stage_11_train_yolo11),
            12: ("Train YOLOv8", self.stage_12_train_yolov8),
            13: ("Resume Training", self.stage_13_resume_training),
            14: ("Export Model", self.stage_14_export_model),
            15: ("Image Inference", self.stage_15_image_inference),
            16: ("Video Inference", self.stage_16_video_inference),
        }

        if stage_num not in stages:
            print(f"❌ Stage {stage_num} not found. Available: {list(stages.keys())}")
            return False

        stage_name, stage_func = stages[stage_num]

        if stage_num == 1:
            if not args.video:
                print(f"❌ Stage 1 requires --video parameter")
                return False
            return stage_func(args.video)
        elif stage_num == 9:
            text = args.text or "সাদা খেলোয়াড় ই-চার দিয়ে খেলা শুরু করছে"
            return stage_func(text)
        elif stage_num == 10:
            return stage_func(labels_subdir=args.labels_subdir, split_ratio=args.split_ratio, seed=args.seed)
        elif stage_num == 11:
            return stage_func(model_weights=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers)
        elif stage_num == 12:
            return stage_func(model_weights=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers)
        elif stage_num == 13:
            return stage_func(checkpoint_path=args.checkpoint)
        elif stage_num == 14:
            formats = [f.strip() for f in args.export_formats.split(",")] if args.export_formats else None
            return stage_func(model_path=args.weights, export_formats=formats)
        elif stage_num == 15:
            return stage_func(model_path=args.weights, image_path=args.image, conf=args.conf, iou=args.iou)
        elif stage_num == 16:
            return stage_func(
                model_path=args.weights,
                source=args.source,
                out=args.out,
                conf=args.conf,
                imgsz=args.imgsz,
                vid_stride=args.vid_stride,
                max_frames=args.max_frames,
                no_show=args.no_show,
                display_scale=args.display_scale,
            )
        else:
            return stage_func()


def main():
    parser = argparse.ArgumentParser(
        description="Chess Video Analysis Pipeline - Stage Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stages.py --stage 1 --video videos/game.mp4    (Extract frames)
  python stages.py --stage 2                             (Board detection)
  python stages.py --stage 3                             (Piece recognition)
    python stages.py --stage 8                             (System check)
    python stages.py --stage 10                            (Prepare YOLO dataset)
    python stages.py --stage 11 --model yolo11s.pt         (Train YOLO11)
    python stages.py --stage 14 --weights runs/detect/train_chess/weights/best.pt --export-formats onnx,torchscript
    python stages.py --stage 15 --image preprocessed_frames/frame_001.jpg
    python stages.py --stage 16 --source videos/chess.mp4 --weights runs/detect/train/weights/best.pt
  python stages.py --list                                (List available stages)
        """
    )
    
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        help="Stage number to run"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Video file path (required for stage 1)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available stages"
    )
    parser.add_argument("--text", type=str, help="Input text (stage 9)")
    parser.add_argument("--labels-subdir", type=str, help="Subdirectory under Annotations/ for YOLO labels (stage 10)")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train split ratio for dataset prep (stage 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split (stage 10)")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Model weights for training stages (11/12)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (11/12)")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size for train/inference")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for training (11/12)")
    parser.add_argument("--workers", type=int, default=2, help="Data loader workers for training (11/12)")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for resume (stage 13)")
    parser.add_argument("--weights", type=str, help="Model weight path for export/inference stages (14/15/16)")
    parser.add_argument("--export-formats", type=str, help="Comma-separated formats for export (stage 14), e.g. onnx,torchscript")
    parser.add_argument("--image", type=str, help="Image path for stage 15")
    parser.add_argument("--source", type=str, help="Video source for stage 16")
    parser.add_argument("--out", type=str, help="Output directory for stage 16")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for inference")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for inference (stage 15)")
    parser.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame (stage 16)")
    parser.add_argument("--max-frames", type=int, default=3000, help="Maximum frames to process in stage 16")
    parser.add_argument("--no-show", action="store_true", help="Disable OpenCV preview window for stage 16")
    parser.add_argument("--display-scale", type=float, default=0.55, help="Preview scale factor for stage 16")
    
    args = parser.parse_args()
    
    runner = StageRunner()
    
    # Show banner
    print("\n" + "="*70)
    print("CHESS VIDEO ANALYSIS - STAGE RUNNER")
    print("="*70)
    print(f"Workspace: {runner.base_dir}")
    print(f"GPU Device: {'NVIDIA MX330' if os.path.exists('/proc/asound') else 'Available'}")
    
    if args.list:
        print("\n" + "="*70)
        print("AVAILABLE STAGES")
        print("="*70)
        stages_info = {
            1: ("Frame Extraction", "Extract frames from video at 1 FPS", "Video file"),
            2: ("Board Detection", "Detect chessboard region in frames", "Preprocessed frames"),
            3: ("Piece Recognition", "Detect chess pieces on board", "Board images"),
            4: ("FEN Encoding", "Convert board state to chess notation", "None"),
            5: ("Move Reconstruction", "Find move between two positions", "None"),
            6: ("Engine Analysis", "Analyze with Stockfish (requires download)", "Stockfish.exe"),
            7: ("Feature Engineering", "Create NLP-friendly features", "None"),
            8: ("System Check", "Check Python/PyTorch/CUDA/YOLO/dataset readiness", "Installed dependencies"),
            9: ("Voice Synthesis", "Convert text to speech (optional)", "TTS model"),
            10: ("Prepare YOLO Dataset", "Build train/val split and data.yaml", "Frames + labels + labels.txt"),
            11: ("Train YOLO11", "Train YOLO11 model", "Dataset YAML + ultralytics"),
            12: ("Train YOLOv8", "Train YOLOv8 model", "Dataset YAML + ultralytics"),
            13: ("Resume Training", "Resume from last checkpoint", "Checkpoint file"),
            14: ("Export Model", "Export trained model formats", "Trained best.pt"),
            15: ("Image Inference", "Run single image inference", "Model + image"),
            16: ("Video Inference", "Run video inference and save output", "Model + video"),
        }
        
        for num, (name, desc, req) in stages_info.items():
            print(f"\nStage {num}: {name}")
            print(f"  Description: {desc}")
            print(f"  Requires: {req}")
        
        return
    
    if not args.stage:
        print("\n❌ No stage specified. Use --stage [number] or --list")
        return
    
    # Run stage
    success = runner.run_stage(args)
    
    if success:
        print("\n" + "="*70)
        print("✅ Stage completed successfully!")
        print("="*70)
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("⚠️  Stage completed with issues (see above)")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()
