#include<stdio.h>
#include<stdlib.h>
#include<time.h>

__global__ void d_matrix_multi(float * d_matrix_a_ptr_in, float * d_matrix_b_ptr_in, float * d_matrix_c_ptr_in, int d_row_size_a_in, int d_row_size_b_in, int d_column_size_c_in)
{
	int d_i=blockIdx.x*blockDim.x+threadIdx.x;
	int d_j=blockIdx.y*blockDim.y+threadIdx.y;
	int d_k;
	float d_sum=(float)0.0;

	for (d_k=0;d_k<d_row_size_b_in;d_k++)
	{
		d_sum+=d_matrix_a_ptr_in[d_i*d_row_size_b_in+d_k]*d_matrix_b_ptr_in[d_k*d_column_size_c_in+d_j];		
	}
		d_matrix_c_ptr_in[d_i*d_column_size_c_in+d_j]=d_sum;
}

int h_matrix_init(float * h_matrix_ptr_in, int h_row_size_in, int h_column_size_in)
{
	int h_i, h_j;

	if(h_matrix_ptr_in==NULL)
	{
		fprintf(stderr, "INVALID MATRIX POINTER.\n");
		return 1;
	}
	else
	{
		srand((unsigned)time(NULL)); 
		for(h_i=0; h_i<h_row_size_in;h_i++)
		{
			for(h_j=0;h_j<h_column_size_in;h_j++)
			{
				h_matrix_ptr_in[h_i*h_column_size_in+h_j]=((float) rand()/(float) RAND_MAX);
			}
		}
		return 0;
	}

}

int h_display_result(float * h_matrix_ptr_in, int h_row_size_in, int h_column_size_in)
{
	int h_i, h_j;

	if (h_matrix_ptr_in==NULL)
	{
		fprintf(stderr,"ERROR IN MATRIX POINTER INPUT.\n");
	   	return 1;
	}
	else if(h_row_size_in==0 || h_column_size_in==0)	
	{
		fprintf(stderr, "ERROR IN MATRIX SIZE INPUT.\n");
		return 1;
	}
	else
	{
		for(h_i=0;h_i<h_row_size_in;h_i++)
		{
			for(h_j=0;h_j<h_column_size_in;h_j++)
			{
				fprintf(stdout,"C[%d][%d]=%f.\n",h_i,h_j,h_matrix_ptr_in[h_i*h_column_size_in+h_j]);
			}
		}

		return 0;
	}
}

int main(int argc, char **argv)
{
	int h_row_size_a, h_row_size_b, h_row_size_c;
	int h_column_size_a, h_column_size_b, h_column_size_c;

	float * h_matrix_a_ptr;
	float * h_matrix_b_ptr;
	float * h_matrix_c_ptr;

	int h_ret=0;

	if(argc!=5)
	{
		fprintf(stderr, "ERROR IN USAGE.\n");
		fprintf(stderr,"./matrix row_size_a column_size_a row_size_b column_size_b \n");
		return 1;
	}
	else
	{	
		h_row_size_a=atoi(argv[1]);
		h_column_size_a=atoi(argv[2]);
		h_row_size_b=atoi(argv[3]);
		h_column_size_b=atoi(argv[4]);

		if((h_row_size_a==0 || h_column_size_a==0 || h_row_size_b==0 || h_column_size_b==0) || (h_column_size_a!=h_row_size_b))
		{
			fprintf(stderr, "INVAILD MATRIX SIZE.\n");
			fprintf(stderr, "C=AxB.\n");
			fprintf(stderr, "Dim for Matrix A is %d x %d.\n", h_row_size_a,h_column_size_a);
			fprintf(stderr, "Dim for Matrix b is %d x %d.\n", h_row_size_b,h_column_size_b);
			return 1;
		}
		else
		{	
			//MATRIX SIZE C
			h_row_size_c=h_row_size_a;
			h_column_size_c=h_column_size_b;

			//UNIFIED MEMORY
			cudaMallocManaged(&h_matrix_a_ptr,h_row_size_a*h_column_size_a*sizeof(float));			}
			cudaMallocManaged(&h_matrix_b_ptr,h_row_size_b*h_column_size_b*sizeof(float));
			cudaMallocManaged(&h_matrix_c_ptr,h_row_size_c*h_column_size_c*sizeof(float));

			//HOST MATRIX INITIALIZATION
			h_ret=h_matrix_init((float *)h_matrix_a_ptr,h_row_size_a, h_column_size_a);
			if(h_ret!=0)
			{
				fprintf(stderr, "MATRIX A INITIALIZATION ERROR.\n");
				return 1;
			}
			h_ret=h_matrix_init((float *)h_matrix_b_ptr, h_row_size_b,h_column_size_b);
			if(h_ret!=0)
			{
				fprintf(stderr, "MATRIX B INITIALIZATION ERROR.\n");
				return 1;
			}
			
		}

		
		//CUDA KERNEL CALL
		dim3 threadsPerBlock(16,16);
		dim3 numBlocks(h_row_size_c/16, h_column_size_c/16);
		d_matrix_multi<<<numBlocks, threadsPerBlock>>>(h_matrix_a_ptr,h_matrix_b_ptr, h_matrix_c_ptr, h_row_size_a, h_column_size_b, h_column_size_c);
		
		//DEVICE SYNCHRONIZATION
		cudaDeviceSynchronize();
		
		//DISPLAY RESULT
		h_ret=h_display_result((float *)h_matrix_c_ptr,h_row_size_c, h_column_size_c);

		//free
		cudaFree(h_matrix_a_ptr);
		cudaFree(h_matrix_b_ptr);
		cudaFree(h_matrix_c_ptr);
	}
	return 0;
}
