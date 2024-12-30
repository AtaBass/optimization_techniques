#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define LEARNING_RATE 0.05
#define ITERATIONS 100
#define NUM_IMAGES 2000
#define NUM_WEIGHTS 5
#define ZERO_TEST_FOLDER "zerotest"
#define FOUR_TEST_FOLDER "fourtest"
#define TEST_IMAGE_COUNT 500
#define MINI_BATCH_SIZE 32

void random_for_w(float *w, int size) {
    int i;
	for (i = 0; i < size; i++) {
        w[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
}



void log_to_file(const char *filename, int iteration, float cost, float time_elapsed) {
    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        printf("!Cannot Open The File! File Name: %s\n", filename);
        exit(-1);
    }
    fprintf(file, "%d %.6f %.6f\n", iteration, cost, time_elapsed);
    fclose(file);
}





void save_weights_with_iterations_to_file(const char *filename, float *weights, int size, int iteration) {
    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        printf("!Cannot Open The File! File Name: %s\n", filename);
        exit(-1);
    }
    fprintf(file, "Iteration %d: ", iteration);
    int i;
	for (i = 0; i < size; i++) {
        fprintf(file, "%.6f ", weights[i]);
    }
    fprintf(file, "\n");
    fclose(file);
}



float* process_image(const char *filename, int *vector_size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("!Cannot Open The File! File Name: %s\n", filename);
        return NULL;
    }

    unsigned char header[54];
    fread(header, sizeof(unsigned char), 54, file);

    int width = *(int *)&header[18];
    int height = *(int *)&header[22];
    int bitDepth = *(int *)&header[28];

    if (bitDepth <= 8) {
        unsigned char colorTable[1024];
        fread(colorTable, sizeof(unsigned char), 1024, file);
    }

    int imageSize = width * height;
    unsigned char *buf = (unsigned char *)malloc(imageSize);
    fread(buf, sizeof(unsigned char), imageSize, file);
    fclose(file);

    *vector_size = imageSize + 1;
    float *vector = (float *)malloc((*vector_size) * sizeof(float));

	int i;
    for (i = 0; i < imageSize; i++) {
        vector[i] = buf[i] / 255.0;
    }
    vector[imageSize] = 1.0;

    free(buf);
    return vector;
}

void log_image_test(const char *filename, int iteration, float cost, float time_elapsed, float *weights, const char *test_image_path, int vector_size) {
    int dummy_size = vector_size; 
    float *test_vector = process_image(test_image_path, &dummy_size);

    if (test_vector == NULL) {
        printf("Could not process the test image: %s\n", test_image_path);
        return;
    }

    float wx = 0.0;
    int i;
	for (i = 0; i < vector_size; i++) {
        wx += weights[i] * test_vector[i];
    }
   

    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        printf("!Cannot Open The File! File Name: %s\n", filename);
        exit(-1);
    }
    fprintf(file, "%d %.6f %.6f\n", iteration, cost, time_elapsed);
    fclose(file);

    free(test_vector);
}


void gradient_descent(float *w, float **vector_a, int num_a, float **vector_b, int num_b, int size, int iterations, float learning_rate) {
    float *gradient = (float *)malloc(size * sizeof(float));
    clock_t start = clock();
	int iter;
    for (iter = 0; iter <= iterations; iter++) {
        float cost = 0.0;
        int i;
		for (i = 0; i < size; i++) {
            gradient[i] = 0.0;
        }
		int sample;
        for (sample = 0; sample < num_a; sample++) {
            float wx = 0.0;
            for (i = 0; i < size; i++) {
                wx += w[i] * vector_a[sample][i];
            }
            float output = tanh(wx);
            float error = output - 1.0;
            cost += pow(error, 2);
            for (i = 0; i < size; i++) {
                gradient[i] += (1 - output * output) * error * vector_a[sample][i] * 2.0f / (num_a + num_b);
            }
        }

        for (sample = 0; sample < num_b; sample++) {
            float wx = 0.0;
            for (i = 0; i < size; i++) {
                wx += w[i] * vector_b[sample][i];
            }
            float output = tanh(wx);
            float error = output - (-1.0);
            cost += pow(error, 2);
            for (i = 0; i < size; i++) {
                gradient[i] += (1 - output * output) * error * vector_b[sample][i] * 2.0f / (num_a + num_b);
            }
        }

        cost = cost / (num_a + num_b);
        clock_t end = clock();
        float time_elapsed = (float)(end - start) / CLOCKS_PER_SEC;
        log_to_file("gd_results.txt", iter, cost, time_elapsed);

        for (i = 0; i < size; i++) {
            w[i] -= learning_rate * gradient[i];
        }

        if (iter % 10 == 0) {
            printf("GD Iteration: %d, Cost: %.6f\n", iter, cost);
        	
		}
		log_image_test("gd_first_image.txt", iter, cost, time_elapsed, w, "zerotest/zero (1).bmp", size);
		save_weights_with_iterations_to_file("gd_weights_with_iters.txt", w, size, iter);
    }
    free(gradient);
}

void mini_batch_sgd(float *w, float **vector_a, int num_a, float **vector_b, int num_b, int size, int iterations, float learning_rate, int batch_size) {
    float *gradient = (float *)malloc(size * sizeof(float));
    clock_t start = clock();
	int iter;
    for (iter = 0; iter <= iterations; iter++) {
        float cost = 0.0;
        int i;
		for (i = 0; i < size; i++) {
            gradient[i] = 0.0;
        }
        int minibatch;
		int sample;
        for (sample = 0; sample < batch_size / 2; sample++) {
            float wx = 0.0;
            minibatch = (rand() % num_a);
            for (i = 0; i < size; i++) {
                wx += w[i] * vector_a[minibatch][i];
            }
            float output = tanh(wx);
            float error = output - 1.0;
            cost += pow(error, 2);
            for (i = 0; i < size; i++) {
                gradient[i] += (1 - output * output) * error * vector_a[minibatch][i] * 2.0f / (batch_size);
            }
        }

        for (sample = 0; sample < batch_size / 2; sample++) {
            float wx = 0.0;
            minibatch = (rand() % num_b);
            for (i = 0; i < size; i++) {
                wx += w[i] * vector_b[minibatch][i];
            }
            float output = tanh(wx);
            float error = output - (-1.0);
            cost += pow(error, 2);
            for (i = 0; i < size; i++) {
                gradient[i] += (1 - output * output) * error * vector_b[minibatch][i] * 2.0f / (batch_size);
            }
        }

        cost = cost / (batch_size);
        clock_t end = clock();
        float time_elapsed = (float)(end - start) / CLOCKS_PER_SEC;
        log_to_file("sgd_results.txt", iter, cost, time_elapsed);

        for (i = 0; i < size; i++) {
            w[i] -= learning_rate * gradient[i];
        }

        if (iter % 10 == 0) {
            printf("SGD Iteration: %d, Cost: %.6f\n", iter, cost);
        

		}
		log_image_test("sgd_first_image.txt", iter, cost, time_elapsed, w, "zerotest/zero (1).bmp", size);
		save_weights_with_iterations_to_file("sgd_weights_with_iters.txt", w, size, iter);
    }
    free(gradient);
}

void adam_optimizer(float *w, float **vector_a, int num_a, float **vector_b, int num_b, int size, int iterations, float learning_rate) {
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-8;

    float *m = (float *)calloc(size, sizeof(float));
    float *v = (float *)calloc(size, sizeof(float));
    float *gradient = (float *)malloc(size * sizeof(float));
	
	int iter;
    clock_t start = clock();
    for (iter = 0; iter <= iterations; iter++) {
        float cost = 0.0;
        int i;
        for (i = 0; i < size; i++) {
            gradient[i] = 0.0;
        }
		int sample;
        for (sample = 0; sample < num_a; sample++) {
            float wx = 0.0;
            for (i = 0; i < size; i++) {
                wx += w[i] * vector_a[sample][i];
            }
            float output = tanh(wx);
            float error = output - 1.0;
            cost += pow(error, 2);
            for (i = 0; i < size; i++) {
                gradient[i] += (1 - output * output) * error * vector_a[sample][i] * 2.0f / (num_a + num_b);
            }
        }

        for (sample = 0; sample < num_b; sample++) {
            float wx = 0.0;
            for (i = 0; i < size; i++) {
                wx += w[i] * vector_b[sample][i];
            }
            float output = tanh(wx);
            float error = output - (-1.0);
            cost += pow(error, 2);
            for (i = 0; i < size; i++) {
                gradient[i] += (1 - output * output) * error * vector_b[sample][i] * 2.0f / (num_a + num_b);
            }
        }

        for (i = 0; i < size; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * gradient[i];
            v[i] = beta2 * v[i] + (1 - beta2) * gradient[i] * gradient[i];

            float m_hat = m[i] / (1 - pow(beta1, iter + 1));
            float v_hat = v[i] / (1 - pow(beta2, iter + 1));

            w[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }

        cost = cost / (num_a + num_b);
        clock_t end = clock();
        float time_elapsed = (float)(end - start) / CLOCKS_PER_SEC;
        log_to_file("adam_results.txt", iter, cost, time_elapsed);
        save_weights_with_iterations_to_file("adam_weights_with_iters.txt", w, size, iter);

        if (iter % 10 == 0) {
            printf("Adam Iteration: %d, Cost: %.6f\n", iter, cost);
        	

		}
        log_image_test("adam_first_image.txt", iter, cost, time_elapsed, w, "zerotest/zero (1).bmp", size);
	}

    free(m);
    free(v);
    free(gradient);
}







void calculate_average_weights(float **weights, float *average_weights, int num_weights, int vector_size) {
    int i,j;
	for (i = 0; i < vector_size; i++) {
        average_weights[i] = 0.0;
        for (j = 0; j < num_weights; j++) {
            average_weights[i] += weights[j][i];
        }
        average_weights[i] /= num_weights;
    }
}

int classify_images(const char *folder_path, float *weights, int vector_size, const char *label, int expected_label) {
    int correct_predictions = 0;
	int img;
    printf("Processing folder: %s\n", folder_path);
    for (img = 1; img <= TEST_IMAGE_COUNT; img++) {
        char filepath[512];
        snprintf(filepath, sizeof(filepath), "%s/%s (%d).bmp", folder_path, label, img);

        float *test_vector = process_image(filepath, &vector_size);
        if (test_vector) {
            float wx = 0.0;
            int i;
            for (i = 0; i < vector_size; i++) {
                wx += weights[i] * test_vector[i];
            }
            int prediction;
			if(tanh(wx) > 0){
				prediction = 1;	
			}
			else{
				prediction = -1;
			}

            if (prediction == expected_label) {
                correct_predictions++;
            }
            free(test_vector);
        }
    }
    return correct_predictions;
}






int main() {
    srand(time(0));
    printf("Model Training Started.\n");
    int vector_size = 0;
    float **vector_a = (float **)malloc(NUM_IMAGES * sizeof(float *));
    float **vector_b = (float **)malloc(NUM_IMAGES * sizeof(float *));
    char filename[256];
	int img;
	
    for (img = 0; img < NUM_IMAGES; img++) {
        sprintf(filename, "zero_train/zero (%d).bmp", img + 1);
        vector_a[img] = process_image(filename, &vector_size);
    }

    for (img = 0; img < NUM_IMAGES; img++) {
        sprintf(filename, "four_train/four (%d).bmp", img + 1);
        vector_b[img] = process_image(filename, &vector_size);
    }

    float **weights_set_gd = (float **)malloc(NUM_WEIGHTS * sizeof(float *));
    float **weights_set_sgd = (float **)malloc(NUM_WEIGHTS * sizeof(float *));
    float **weights_set_adam = (float **)malloc(NUM_WEIGHTS * sizeof(float *));

	int i;
    for (i = 0; i < NUM_WEIGHTS; i++) {
        weights_set_gd[i] = (float *)malloc(vector_size * sizeof(float));
        random_for_w(weights_set_gd[i], vector_size);

        printf("\nGD Training for Weight Set %d\n", i + 1);
        gradient_descent(weights_set_gd[i], vector_a, NUM_IMAGES, vector_b, NUM_IMAGES, vector_size, ITERATIONS, LEARNING_RATE);
    }

    for (i = 0; i < NUM_WEIGHTS; i++) {
        weights_set_sgd[i] = (float *)malloc(vector_size * sizeof(float));
        random_for_w(weights_set_sgd[i], vector_size);

        printf("\nMini-Batch SGD Training for Weight Set %d\n", i + 1);
        mini_batch_sgd(weights_set_sgd[i], vector_a, NUM_IMAGES, vector_b, NUM_IMAGES, vector_size, 200, LEARNING_RATE, MINI_BATCH_SIZE);
    }

    for (i = 0; i < NUM_WEIGHTS; i++) {
        weights_set_adam[i] = (float *)malloc(vector_size * sizeof(float));
        random_for_w(weights_set_adam[i], vector_size);

        printf("\nAdam Training for Weight Set %d\n", i + 1);
        adam_optimizer(weights_set_adam[i], vector_a, NUM_IMAGES, vector_b, NUM_IMAGES, vector_size, ITERATIONS, LEARNING_RATE);
    }

    float *average_weights_gd = (float *)malloc(vector_size * sizeof(float));
    float *average_weights_sgd = (float *)malloc(vector_size * sizeof(float));
    float *average_weights_adam = (float *)malloc(vector_size * sizeof(float));

    calculate_average_weights(weights_set_gd, average_weights_gd, NUM_WEIGHTS, vector_size);
    calculate_average_weights(weights_set_sgd, average_weights_sgd, NUM_WEIGHTS, vector_size);
    calculate_average_weights(weights_set_adam, average_weights_adam, NUM_WEIGHTS, vector_size);

    printf("\nEvaluating GD Weights:\n");
    int correct_zero_gd = classify_images(ZERO_TEST_FOLDER, average_weights_gd, vector_size, "zero", 1);
    int correct_four_gd = classify_images(FOUR_TEST_FOLDER, average_weights_gd, vector_size, "four", -1);

    

    printf("\nEvaluating Mini-Batch SGD Weights:\n");
    int correct_zero_sgd = classify_images(ZERO_TEST_FOLDER, average_weights_sgd, vector_size, "zero", 1);
    int correct_four_sgd = classify_images(FOUR_TEST_FOLDER, average_weights_sgd, vector_size, "four", -1);

    

    printf("\nEvaluating Adam Weights:\n");
    int correct_zero_adam = classify_images(ZERO_TEST_FOLDER, average_weights_adam, vector_size, "zero", 1);
    int correct_four_adam = classify_images(FOUR_TEST_FOLDER, average_weights_adam, vector_size, "four", -1);

    

    int total_test_images = TEST_IMAGE_COUNT * 2;

    printf("\nGD Total Correct Predictions: %d/%d\n", correct_zero_gd + correct_four_gd, total_test_images);
    printf("GD Accuracy: %.2f%%\n", ((correct_zero_gd + correct_four_gd) / (float)total_test_images) * 100);

    printf("\nMini-Batch SGD Total Correct Predictions: %d/%d\n", correct_zero_sgd + correct_four_sgd, total_test_images);
    printf("Mini-Batch SGD Accuracy: %.2f%%\n", ((correct_zero_sgd + correct_four_sgd) / (float)total_test_images) * 100);

    printf("\nAdam Total Correct Predictions: %d/%d\n", correct_zero_adam + correct_four_adam, total_test_images);
    printf("Adam Accuracy: %.2f%%\n", ((correct_zero_adam + correct_four_adam) / (float)total_test_images) * 100);

    for (img = 0; img < NUM_IMAGES; img++) {
        free(vector_a[img]);
        free(vector_b[img]);
    }
    free(vector_a);
    free(vector_b);
    
	for (i = 0; i < NUM_WEIGHTS; i++) {
        free(weights_set_gd[i]);
        free(weights_set_sgd[i]);
        free(weights_set_adam[i]);
    }
    free(weights_set_gd);
    free(weights_set_sgd);
    free(weights_set_adam);
    free(average_weights_gd);
    free(average_weights_sgd);
    free(average_weights_adam);
    
	return 0;
}
