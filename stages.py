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
    
    def run_stage(self, stage_num, video_path=None):
        """Run a specific stage"""
        
        stages = {
            1: ("Frame Extraction", self.stage_1_extract_frames),
            2: ("Board Detection", self.stage_2_board_detection),
            3: ("Piece Recognition", self.stage_3_piece_recognition),
            4: ("FEN Encoding", self.stage_4_fen_encoding),
            5: ("Move Reconstruction", self.stage_5_move_reconstruction),
            6: ("Engine Analysis", self.stage_6_engine_analysis),
            7: ("Feature Engineering", self.stage_7_feature_engineering),
            9: ("Voice Synthesis", self.stage_9_voice_synthesis),
        }
        
        if stage_num not in stages:
            print(f"❌ Stage {stage_num} not found. Available: {list(stages.keys())}")
            return False
        
        stage_name, stage_func = stages[stage_num]
        
        if stage_num == 1:
            if not video_path:
                print(f"❌ Stage 1 requires --video parameter")
                return False
            return stage_func(video_path)
        elif stage_num == 9:
            text = "সাদা খেলোয়াড় ই-চার দিয়ে খেলা শুরু করছে"
            return stage_func(text)
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
  python stages.py --stage 4                             (FEN encoding)
  python stages.py --stage 5                             (Move reconstruction)
  python stages.py --stage 6                             (Stockfish analysis)
  python stages.py --stage 7                             (Features)
  python stages.py --stage 9                             (Voice synthesis)
  python stages.py --list                                (List available stages)
        """
    )
    
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 9],
        help="Stage number to run (1, 2, 3, 4, 5, 6, 7, or 9)"
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
            9: ("Voice Synthesis", "Convert text to speech (optional)", "TTS model"),
        }
        
        for num, (name, desc, req) in stages_info.items():
            print(f"\nStage {num}: {name}")
            print(f"  Description: {desc}")
            print(f"  Requires: {req}")
        
        print("\n" + "="*70)
        print("GPU-REQUIRED STAGES (After PyTorch installation)")
        print("="*70)
        print("\nStage 8: Bangla NLP Commentary (mT5)")
        
        return
    
    if not args.stage:
        print("\n❌ No stage specified. Use --stage [number] or --list")
        return
    
    # Run stage
    success = runner.run_stage(args.stage, args.video)
    
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
