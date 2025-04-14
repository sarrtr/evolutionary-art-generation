# Evolutionary Art Generation

This project is a visual evolutionary algorithm that creates unique abstract images through genetic evolution. By using a mix of shapes, colors, and randomness, it evolves images over multiple generations based on user selection and internal diversity. It’s a creative blend of genetic algorithms and generative art, offering an engaging way to explore visual evolution interactively.

## How to run
1. Create virtual environment
    (e.g., ``` python -m venv venv ```)
2. Activate virtual environment
    - On Windows: ``` venv\Scripts\activate ```
    - On macOS/Linux: ``` source venv/bin/activate ```
4. Run ``` pip install -r requirements.txt ```
5. Run ``` python app.py ```

## Program Overview

At each generation, a grid of generated images will appear. You can click to select the one you like most—or select nothing at all. Once a choice is made (or skipped), the next generation will be created based on your input. A progress bar at the bottom of the screen shows how close you are to generating the next batch of evolved images.

![alt text](assets/progress_bar.gif)

## Functionality

Here are all the interactive controls:

- Reset – Starts a new evolution from scratch. This will discard all current progress and begin with a fresh population.

- Stop evolution – Finishes the evolution and terminates the program.

- Choose next - Skips choice of the image.

- Image selection – Click on any image to indicate it as your preferred choice for guiding the next generation*.

**Note:**  
Every time the program starts, the history/ folder is automatically cleared. This means any previously generated images will be permanently deleted. If you want to keep your favorite images, be sure to save them manually before exiting the program.

## Project Structure
```
evolutionary-art-generation/
├── app.py                  # Main entry point (runs the app)
├── genetic_algorithm.py    # Core logic for image creation and evolution
├── assets                  # README assets            
├── requirements.txt        # Dependencies
└── README.md               # You're here!
```
