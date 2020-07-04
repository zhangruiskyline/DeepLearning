You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols + and -. For each integer, you should choose one from + and - as its new symbol.

 

Find out how many ways to assign symbols to make sum of integers equal to target S.

Example 1:

Input: nums is [1, 1, 1, 1, 1], S is 3. 

Output: 5

Explanation: 

 

-1+1+1+1+1 = 3

+1-1+1+1+1 = 3

+1+1-1+1+1 = 3

+1+1+1-1+1 = 3

+1+1+1+1-1 = 3

 

There are 5 ways to assign symbols to make the sum of nums be target 3.

Note:

The length of the given array is positive and will not exceed 20.

The sum of elements in the given array will not exceed 1000.

Your output answer is guaranteed to be fitted in a 32-bit integer.


ObjA = new share_pointer();

```CPP


int sumSymbol(vector<uint> in, int target){
	int ret = 0;
	help(in,target,ret,0,0);

	return ret;

}

void help(vector<uint> &in, int target, int &ret, int val, int index){

	if(val==target && index == in.size()-1){
		ret++;
		return;
	}

	if(index == in.size()-1){
		return;
	}

	for(int i =0; i< in.size(); i++){
		help(in, target, ret, val+in[i], index+1);
		help(in, target, ret, val-in[i], index+1);
	}

}



int sumSymbol(vector<uint> in, int target){
	int ret = 0;
	//row: index,
	//col: all possible sum,
	//val: occurence
	std::vector<vector<int>> dp;
	std::vector<int> one;

	for(int i=0; i<in.size(); i++){

		dp[i][j] = dp[i-1][j+in[i]] + dp[i-1][j-in[i]];


	}

	for(int j=0;j<one.size();j++){
		if(dp[in.size()-1][j]==target){
			ret++;
		}
	}

	return ret;



}


```