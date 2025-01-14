#include <SDL.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <random>

using namespace std;

//Random number generator
unsigned seed = chrono::system_clock::now().time_since_epoch() / chrono::microseconds(1) % 1000000;
default_random_engine rnd(seed);
uniform_real_distribution<float> rndFloat(-1.0, 1.0);

//Window parameters
const int FPS = 100;
const int GRID_WIDTH = 200;
const int GRID_HEIGHT = 200;
const int PIXEL_SIZE = 3;

//Simulation parameters
bool CONTINUE = false;   //
bool SAVE = false;       //TODO: modify saving and continuing for variable number of inner layers
bool VISUALIZE = true;
const int GENERATION_LIFETIME = 200;
const int ORGANISM_INPUT_SIZE = 10;
const int ORGANISM_INNER_SIZE = 16;
const int ORGANISM_OUTPUT_SIZE = 8;

//Initializing array with random update order
pair <int, int> upd_order[GRID_WIDTH * GRID_HEIGHT];

//Inner variables
long long tick_cap = int(1000 / FPS);
long long start, tick_time;
int WINDOW_WIDTH = GRID_WIDTH * PIXEL_SIZE;
int WINDOW_HEIGHT = GRID_HEIGHT * PIXEL_SIZE;


//Functions
int randSign() {
	if (rnd() % 2 == 0) {
		return 1;
	}
	else {
		return -1;
	}
}
bool randBool() {
	if (rnd() % 2 == 0) {
		return true;
	}
	else {
		return false;
	}
}
void drawCircle(SDL_Renderer* rend, int x, int y, int rad, int r, int g, int b) {
	for (int i = x - rad - 1; i != x + rad + 1; i++) {
		for (int j = y - rad - 1; j != y + rad + 1; j++) {
			if ((x - i) * (x - i) + (y - j) * (y - j) <= rad * rad) {
				SDL_SetRenderDrawColor(rend, r, g, b, 255);
				SDL_RenderDrawPoint(rend, i, j);
				SDL_SetRenderDrawColor(rend, 255, 255, 255, 255);
			}
		}
	}
}


//Structures
struct Layer {
	int inS, outS;
	float** weights;
	float* biases;

	Layer(int inputSize, int outputSize) {
		inS = inputSize;
		outS = outputSize;

		weights = new float* [outS]; 
		for (int i = 0; i < outS; i++) {
			weights[i] = new float[inS];
			for (int j = 0; j < inS; j++) {
				weights[i][j] = 0.0;
			}
		}

		biases = new float [outS];
		for (int i = 0; i < outS; i++) {
			biases[i] = 0.0;
		}
	}

	~Layer() {
		for (int i = 0; i < outS; i++) {
			delete[] weights[i];
		}
		delete[] weights;
		delete[] biases;
	}

	void SetRandW() {
		for (int i = 0; i < outS; i++) {
			for (int j = 0; j < inS; j++) {
				weights[i][j] = rndFloat(rnd);
			}
		}
	}
	void SetRandB() {
		for (int i = 0; i < outS; i++) {
			biases[i] = rndFloat(rnd);
		}
	}
	float ReLU(float x) {
		if (x > 0) {
			return x;
		}
		else {
			return 0.0;
		}
	}
	float* Process(float* inArr) {
		float sum;
		float* outArr = new float[outS];
		for (int i = 0; i < outS; i++) {
			sum = biases[i];
			for (int j = 0; j < inS; j++) {
				sum += inArr[j] * weights[i][j];
			}
			outArr[i] = ReLU(sum);
		}
		return outArr;
	}

};
struct Organism {
	int infoS, innerS, decS;
	Layer* layer1;
	Layer* layer2;

	Organism(int inS, int midS, int outS): infoS(inS), innerS(midS), decS(outS) {
		layer1 = new Layer(infoS, midS);
		layer1->SetRandW();
		layer1->SetRandB();
		
		layer2 = new Layer(midS, decS);
		layer2->SetRandW();
		layer2->SetRandB();
	}
	~Organism() {
		delete layer1;
		delete layer2;
	}

	void Draw(SDL_Renderer* rend, int x, int y) {
		SDL_SetRenderDrawColor(rend, 0, 191, 255, 255);
		for (int i = 0; i != PIXEL_SIZE; i++) {
			for (int j = 0; j != PIXEL_SIZE; j++) {
				SDL_RenderDrawPoint(rend, x * PIXEL_SIZE + i, y * PIXEL_SIZE + j);
			}
		}
	}
	int GetMaxId(float* arr, int len) {
		if (len < 2) {
			return 0;
		}
		int id = 0;
		float max = arr[0];
		for (int i = 1; i != len; i++) {
			if (arr[i] > max) {
				max = arr[i];
				id = i;
			}
		}
		return id;
	}
	int Poll(float* infoArr) {
		float* layerOut1 = layer1->Process(infoArr);
		float* decisionArr = layer2->Process(layerOut1);

		int decisionId = GetMaxId(decisionArr, decS);
		delete[] decisionArr;
		return decisionId;
	}
	Organism* Clone(float mutationFactor = 0.0) {
		Organism* clone = new Organism(infoS, innerS, decS);

		for (int i = 0; i < innerS; i++) {
			for (int j = 0; j < infoS; j++) {
				clone->layer1->weights[i][j] = this->layer1->weights[i][j] * (1.0 + mutationFactor * randSign());
			}
			clone->layer1->biases[i] = this->layer1->biases[i] * (1.0 + mutationFactor * randSign());
		}

		for (int i = 0; i < decS; i++) {
			for (int j = 0; j < innerS; j++) {
				clone->layer2->weights[i][j] = this->layer2->weights[i][j] * (1.0 + mutationFactor * randSign());
			}
			clone->layer2->biases[i] = this->layer2->biases[i] * (1.0 + mutationFactor * randSign());
		}

		return clone;
	}

};


//Grid structure
struct Grid {
	Organism*** cells;
	int width, height;
	int xCrit, yCrit;
	pair <int, int>* ord;

	Grid(int w, int h, pair <int, int>* arr) {
		width = w;
		height = h;
		xCrit = rnd() % width;
		yCrit = rnd() % height;
		ord = arr;
		cells = new Organism** [width];
		for (int i = 0; i < width; i++)
			cells[i] = new Organism* [height];
	}
	~Grid() {
		for (int x = width - 1; x != -1; x++) {
			for (int y = height - 1; y != -1; y++) {
				if (cells[x][y]) {
					delete cells[x][y];
				}
			}
			delete[] cells[x];
		}
		delete[] cells;
	}

	void Draw(SDL_Renderer* rend) {
		for (int x = 0; x != width; x++) {
			for (int y = 0; y != height; y++) {
				if (cells[x][y]) {
					cells[x][y]->Draw(rend, x, y);
				}
				else if (FitsCriteria(x, y)) {
					SDL_SetRenderDrawColor(rend, 144, 238, 144, 255);
					for (int i = 0; i != PIXEL_SIZE; i++) {
						for (int j = 0; j != PIXEL_SIZE; j++) {
							SDL_RenderDrawPoint(rend, x * PIXEL_SIZE + i, y * PIXEL_SIZE + j);
						}
					}
				}
			}
		}
	}
	void Fill(SDL_Renderer* rend, int r, int g, int b) {
		SDL_SetRenderDrawColor(rend, r, g, b, 255);
		SDL_RenderClear(rend);
	}
	bool IsEmpty(int x, int y) {
		if (x < 0 or x >= width or y < 0 or y >= height) {
			return false;
		}
		else if (cells[x][y] == nullptr) {
			return true;
		}
		return false;
	}
	float GetVal(int x, int y) {
		if (x < 0 or x >= width or y < 0 or y >= height) {
			return -1.0;
		}
		else if (cells[x][y] == nullptr) {
			return 0.0;
		}
		else {
			return 1.0;
		}
	}
	void MoveFromTo(int x0, int y0, int x1, int y1) {
		Organism* temp = cells[x0][y0];
		cells[x0][y0] = nullptr;
		cells[x1][y1] = temp;
	}
	void ChangeUpdOrder() {
		shuffle(ord, ord + width * height, rnd);
	}
	bool FitsCriteria(int x, int y) {
		if (abs(x - xCrit) < width * 0.05 and abs(y - yCrit) < height * 0.05) {
			return true;
		}
		return false;
	}
	void ApplySelection() {
		vector<Organism*> survivors;
		vector<Organism*> fallback;
		int initN = 0;
		int survN = 0;

		for (int x = 0; x != width; x++) {
			for (int y = 0; y != height; y++) {
				if (cells[x][y]) {
					initN++;
					fallback.push_back(cells[x][y]->Clone(0.5f));
					if (FitsCriteria(x, y)) {
						survivors.push_back(cells[x][y]);
						survN++;
					}
					else {
						delete cells[x][y];
					}
					cells[x][y] = nullptr;
				}
			}
		}

		if (survivors.empty()) {
			survivors = fallback;
		}
		else {
			for (int i = survN; i != initN; i++) {
				survivors.push_back(survivors[rnd() % survN]->Clone(rndFloat(rnd) / 10.f));
			}
		}

		ChangeUpdOrder();
		for (int i = 0; i != initN; i++) {
			cells[ord[i].first][ord[i].second] = survivors[i];
		}
		ChangeUpdOrder();

		xCrit = rnd() % width;
		yCrit = rnd() % height;
	}
	void CompriseInfoArr(int x, int y, float* infoArr) {
		infoArr[0] = float(xCrit - x) * 2 / width - 1.0;
		infoArr[1] = float(yCrit - y) * 2 / height - 1.0;
		infoArr[2] = GetVal(x - 1, y - 1);
		infoArr[3] = GetVal(x - 1, y);
		infoArr[4] = GetVal(x - 1, y + 1);
		infoArr[5] = GetVal(x, y - 1);
		infoArr[6] = GetVal(x, y + 1);
		infoArr[7] = GetVal(x + 1, y - 1);
		infoArr[8] = GetVal(x + 1, y);
		infoArr[9] = GetVal(x + 1, y + 1);
	}
	void Update() {
		int x, y;
		float infoArr[ORGANISM_INPUT_SIZE] = {0.0};

		for (int i = 0; i != width * height; i++) {
			x = ord[i].first;
			y = ord[i].second;

			if (cells[x][y] == nullptr) {
				continue;
			}


			CompriseInfoArr(x, y, infoArr);

			switch (cells[x][y]->Poll(infoArr)) {

			case 0:
				if (IsEmpty(x - 1, y - 1)) {
					MoveFromTo(x, y, x - 1, y - 1);
				}
				break;
			case 1:
				if (IsEmpty(x, y - 1)) {
					MoveFromTo(x, y, x, y - 1);
				}
				break;
			case 2:
				if (IsEmpty(x + 1, y - 1)) {
					MoveFromTo(x, y, x + 1, y - 1);
				}
				break;
			case 3:
				if (IsEmpty(x + 1, y)) {
					MoveFromTo(x, y, x + 1, y);
				}
				break;
			case 4:
				if (IsEmpty(x + 1, y + 1)) {
					MoveFromTo(x, y, x + 1, y + 1);
				}
				break;
			case 5:
				if (IsEmpty(x, y + 1)) {
					MoveFromTo(x, y, x, y + 1);
				}
				break;
			case 6:
				if (IsEmpty(x - 1, y + 1)) {
					MoveFromTo(x, y, x - 1, y + 1);
				}
				break;
			case 7:
				if (IsEmpty(x - 1, y)) {
					MoveFromTo(x, y, x - 1, y);
				}
				break;
			}
		}
	}

};

int main(int argc, char* args[]) {


	//Filling and shuffling update order array
	int n = 0;
	for (int x = 0; x != GRID_WIDTH; x++) {
		for (int y = 0; y != GRID_HEIGHT; y++) {
			upd_order[n].first = x;
			upd_order[n].second = y;
			n++;
		}
	}
	shuffle(upd_order, upd_order + GRID_WIDTH * GRID_HEIGHT, rnd);


	//Grid initialization
	Grid grid = Grid(GRID_WIDTH, GRID_HEIGHT, upd_order);
	for (int x = 0; x != grid.width; x++) {
		for (int y = 0; y != grid.height; y++) {
				grid.cells[x][y] = nullptr;
		}
	}


	//Organisms initialization
	if (CONTINUE) {

		ifstream ParametersFile("Weights_and_biases.txt");
		string line;
		int x = 0;
		int y = 0;
		int layerLine = 0;
		Organism* tempRead = nullptr;

		for (line; getline(ParametersFile, line);) {

			istringstream in(line);
			string type;
			in >> type;

			if (type == "Coordinates") {
				tempRead = new Organism(ORGANISM_INPUT_SIZE, ORGANISM_INNER_SIZE, ORGANISM_OUTPUT_SIZE);
				in >> x >> y;
			}
			else if (type == "Layer") {
				for (int i = 0; i < tempRead->infoS; i++) {
					in >> tempRead->layer1->weights[layerLine][i];
				}
				in >> tempRead->layer1->biases[layerLine];
				layerLine++;
			}
			else if (type == "OrganismEnd") {
				grid.cells[x][y] = tempRead;
				x = 0;
				y = 0;
				layerLine = 0;
			}

		}

	}
	else {
		for (int x = 0; x != grid.width; x++) {
			for (int y = 0; y != grid.height; y++) {
				if (rnd() % 100 == 0) {
					grid.cells[x][y] = new Organism(ORGANISM_INPUT_SIZE, ORGANISM_INNER_SIZE, ORGANISM_OUTPUT_SIZE);
				}
			}
		}
	}


	//Creating window and renderer
	SDL_Window* window = nullptr;
	SDL_Renderer* rend = nullptr;
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
		return -1;
	}
	window = SDL_CreateWindow("SDL Tutorial", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
	if (window == nullptr) {
		printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
		return -1;
	}
	rend = SDL_CreateRenderer(window, 0, SDL_RENDERER_ACCELERATED);


	//Main loop
	int tickCnt = 0;
	int genCnt = 1;

	cout << "Generation " << genCnt << "\n";

	SDL_Event event;
	bool quit = false;
	while (not quit) {
		if (VISUALIZE) {

			//Write down starting time
			start = chrono::system_clock::now().time_since_epoch() / chrono::milliseconds(1);

			//Check if "close window" button was pressed
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_QUIT) quit = true;
			}

			//Clear previous window
			grid.Fill(rend, 0, 0, 0);

			//Draw whatever is needed
			grid.Draw(rend);

			//Calculate whatever is needed
			grid.ChangeUpdOrder();
			grid.Update();
			tickCnt++;
			if (tickCnt > GENERATION_LIFETIME) {
				tickCnt = 0;
				genCnt++;
				cout << "Generation " << genCnt << "\n";
				grid.ApplySelection();
			}

			//Show updates
			SDL_RenderPresent(rend);

			//Sleep if neccessary
			tick_time = chrono::system_clock::now().time_since_epoch() / chrono::milliseconds(1) - start;
			if (tick_time < tick_cap) {
				SDL_Delay(tick_cap - tick_time);
			}

		}
		else {

			//Check if "close window" button was pressed
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_QUIT) quit = true;
			}

			//Only calculations
			grid.ChangeUpdOrder();
			grid.Update();
			tickCnt++;
			if (tickCnt > GENERATION_LIFETIME) {
				tickCnt = 0;
				genCnt++;
				cout << "Generation " << genCnt << "\n";
				grid.ApplySelection();
			}

		}
	}


	//Destroy window and renderer
	SDL_DestroyWindow(window);
	SDL_DestroyRenderer(rend);
	SDL_Quit();

	//Save parameters if necessary
	if (SAVE) {

		ofstream ParametersFile("Weights_and_biases.txt");
		
		Organism* tempSave;
		for (int x = 0; x != grid.width; x++) {
			for (int y = 0; y != grid.height; y++) {
				tempSave = grid.cells[x][y];
				if (tempSave) {
					ParametersFile << "Coordinates " << x << " " << y << "\n";
					for (int i = 0; i < tempSave->decS; i++) {
						ParametersFile << "Layer ";
						for (int j = 0; j < tempSave->infoS; j++) {
							ParametersFile << tempSave->layer1->weights[i][j] << " ";
						}
						ParametersFile << tempSave->layer1->biases[i] << "\n";
					}
					ParametersFile << "OrganismEnd\n";
				}
			}
		}

		ParametersFile.close();
		cout << "Finished saving" << "\n";
	}

	//Free the heap
	delete& grid;

	return 0;
}