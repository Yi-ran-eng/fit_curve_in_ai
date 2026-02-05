# fit_curve_in_ai
__learn to fit a curve in a machine learning way__

of course that is complex, and ,to be honest, this model function is not really good along with the given constants:
* ${z_{0}=9.7e-3}$
* ${L_{eff}=1.3912e-4}$
* ${I_{0}=5e13}$

these numbers have remarkable differences, and it's not allowed to make a scale mapping about the input datas,what you need to do is to find the property ${\alpha_{NL}}$ in this function.

for me , i use one mapping mean to split integrals into discrete sums, it may be the best way to handle integrations of infinite intervals like this.Then ,you can choose to use the optimizing way enbedded in torch or compute it by yourself,in my experiments i finished it in the last way. for more about these, i computed a function that allows you to run it to get a subclass of one parent class in your code,you can freely create what you want in individual function that defined by yourself and pass it into the subclass.I think this way boosts the flexibility of nn.Module's subclass.

the difference about the two codes is the Jloss function, in the _2 file, i write the Jloss function alone to make the sum of these values which below fitted curve as close as possible to the sum of values which above fitted curve. this way caused some negative value in loss, but that is what we have to do in achieving the goal for it follows the principle of fitting a curve instead of machine learning. in Inffixed file, i use the normal machine learning way ,MSELoss, to calculate the loss.

that's something about these codes, and the puctures give some results about the codes, clearly it doesn't fit very well, and if you read these pictures about certain alphaNL,then you may find the huge error in fitting is raised by param z0,so what i want to say is the model function doesn't works well in such a given constants.
