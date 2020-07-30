#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>

using namespace std;

#define max 10000	// Maximum Configurable Matrix Size

int a[max][max];
int b[max][max];
int c[max][max];
int d[max][max];

void matrix_generation(int n)	// Function to randonly generate matrix elements
{
	int i,j;
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			a[i][j]=rand()%100;
			b[i][j]=rand()%100;
			
		}
	}
}
void serial_multiply_matrices(int n)	// The conventional Matrix multiplication function
{
	int i,j,k;
	double st=omp_get_wtime();
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			for(k=0;k<n;k++)
			{
				c[i][j]+=a[i][k]*b[k][j];
			}
		}
	}
	double en=omp_get_wtime();
	printf("Matrix Multiplication using Conventional Serial Multiplication : %lf Seconds\n",en-st);
}
void parallel_multiply_static(int n)	//Function to multiply matrices using static scheduler
{
	memset(d,0,sizeof d);
	int i,j,k;
	double st=omp_get_wtime();
	#pragma omp parallel for schedule(static,50) collapse(2) private(i,j,k)shared(a,b)
	for(i=0;i<n;i++)for( j=0;j<n;j++)for(k=0;k<n;k++)d[i][j]+=a[i][k]*b[k][j];
	double en=omp_get_wtime();
	printf("Matrix Multiplication using Static Scheduler : %lf Seconds \n",en-st);
}

void parallel_multiply_dynamic(int n)	//Function to multiply matrices using dynamic scheduler 
{
	memset(d,0,sizeof d);
	int i,j,k;
	double st1=omp_get_wtime();
	#pragma omp parallel for schedule(dynamic,50) collapse(2) private(i,j,k) shared(a,b)
	for(i=0;i<n;i++){
		for( j=0;j<n;j++){
			for(k=0;k<n;k++){
				d[i][j]+=a[i][k]*b[k][j];
			}
		}
	}
	double en1=omp_get_wtime();
	printf("Matrix Multiplication using Dynamic Scheduler : %lf Seconds\n",en1-st1);
}
int main() {

	int n;
	cout << "Enter Size of Matrices to be generated and multiplied:";
	
	cin >> n;
	if(n <= 0)
	{
		cout << "Enter Valid Positive Matrix Size!!";
		exit(1);
	}
	
	matrix_generation(n);	
	
	serial_multiply_matrices(n);
	
	parallel_multiply_static(n);
	
	parallel_multiply_dynamic(n);

	return 0;
	
}

