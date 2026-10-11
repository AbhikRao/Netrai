classdef ReduceMeanLayer1003 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end


    methods(Static, Hidden)
        % Specify the path to the class that will be used for codegen
        function name = matlabCodegenRedirect(~)
            name = 'netrai_fold0.coder.ReduceMeanLayer1003';
        end
    end


    methods
        function this = ReduceMeanLayer1003(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone__46'};
        end

        function [x_model_backbone__46] = predict(this, x_model_backbone__40)
            if isdlarray(x_model_backbone__40)
                x_model_backbone__40 = stripdims(x_model_backbone__40);
            end
            x_model_backbone__40NumDims = 4;
            x_model_backbone__40 = netrai_fold0.ops.permuteInputVar(x_model_backbone__40, [4 3 1 2], 4);

            [x_model_backbone__46, x_model_backbone__46NumDims] = ReduceMeanGraph1009(this, x_model_backbone__40, x_model_backbone__40NumDims, false);
            x_model_backbone__46 = netrai_fold0.ops.permuteOutputVar(x_model_backbone__46, [3 4 2 1], 4);

            x_model_backbone__46 = dlarray(single(x_model_backbone__46), 'SSCB');
        end

        function [x_model_backbone__46] = forward(this, x_model_backbone__40)
            if isdlarray(x_model_backbone__40)
                x_model_backbone__40 = stripdims(x_model_backbone__40);
            end
            x_model_backbone__40NumDims = 4;
            x_model_backbone__40 = netrai_fold0.ops.permuteInputVar(x_model_backbone__40, [4 3 1 2], 4);

            [x_model_backbone__46, x_model_backbone__46NumDims] = ReduceMeanGraph1009(this, x_model_backbone__40, x_model_backbone__40NumDims, true);
            x_model_backbone__46 = netrai_fold0.ops.permuteOutputVar(x_model_backbone__46, [3 4 2 1], 4);

            x_model_backbone__46 = dlarray(single(x_model_backbone__46), 'SSCB');
        end

        function [x_model_backbone__46, x_model_backbone__46NumDims1011] = ReduceMeanGraph1009(this, x_model_backbone__40, x_model_backbone__40NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1010, x_model_backbone__40NumDims);
            xMean = mean(x_model_backbone__40, dims);
            x_model_backbone__46 = xMean;
            x_model_backbone__46NumDims = x_model_backbone__40NumDims;

            % Set graph output arguments
            x_model_backbone__46NumDims1011 = x_model_backbone__46NumDims;

        end

    end

end