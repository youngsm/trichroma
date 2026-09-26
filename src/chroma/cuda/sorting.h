template <class T>
__device__ void
swap(T &a, T &b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

template <class T>
__device__ void
reverse(int n, T *a)
{
    for (int i=0; i < n/2; i++)
	swap(a[i],a[n-1-i]);
}

template <class T>
__device__ void
piksrt(int n, T *arr)
{
    int i,j;
    T a;

    for (j=1; j < n; j++) {
	a = arr[j];
	i = j-1;
	while (i >= 0 && arr[i] > a) {
	    arr[i+1] = arr[i];
	    i--;
	}
	arr[i+1] = a;
    }
}

template <class T, class U>
__device__ void
piksrt2(int n, T *arr, U *brr)
{
    int i,j;
    T a;
    U b;

    for (j=1; j < n; j++) {
	a = arr[j];
	b = brr[j];
	i = j-1;
	while (i >= 0 && arr[i] > a) {
	    arr[i+1] = arr[i];
	    brr[i+1] = brr[i];
	    i--;
	}
	arr[i+1] = a;
	brr[i+1] = b;
    }
}

/* Returns the index in `arr` where `x` should be inserted in order to
   maintain order. If `n` equals one, return the index such that, when
   `x` is inserted, `arr` will be in ascending order.
*/
template <class T>
__device__ unsigned long
searchsorted(unsigned long n, T *arr, const T &x)
{
    unsigned long ju,jm,jl;
    int ascnd;

    jl = 0;
    ju = n;

    ascnd = (arr[n-1] >= arr[0]);

    while (ju-jl > 1) {
	jm = (ju+jl) >> 1;

	if ((x > arr[jm]) == ascnd)
	    jl = jm;
	else
	    ju = jm;
    }

    if ((x <= arr[0]) == ascnd)
	return 0;
    else
	return ju;
}

template <class T>
__device__ void
insert(unsigned long n, T *arr, unsigned long i, const T &x)
{
    unsigned long j;
    for (j=n-1; j > i; j--)
	arr[j] = arr[j-1];
    arr[i] = x;
}

template <class T>
__device__ void
add_sorted(unsigned long n, T *arr, const T &x)
{
    unsigned long i = searchsorted(n, arr, x);

    if (i < n)
	insert(n, arr, i, x);
}
