classdef ReduceMeanLayer1028 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1028';
        end
    end


    methods
        function this = ReduceMeanLayer1028(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_417'};
        end

        function [x_model_backbone_417] = predict(this, x_model_backbone_411)
            if isdlarray(x_model_backbone_411)
                x_model_backbone_411 = stripdims(x_model_backbone_411);
            end
            x_model_backbone_411NumDims = 4;
            x_model_backbone_411 = netrai_fold0.ops.permuteInputVar(x_model_backbone_411, [4 3 1 2], 4);

            [x_model_backbone_417, x_model_backbone_417NumDims] = ReduceMeanGraph1084(this, x_model_backbone_411, x_model_backbone_411NumDims, false);
            x_model_backbone_417 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_417, [3 4 2 1], 4);

            x_model_backbone_417 = dlarray(single(x_model_backbone_417), 'SSCB');
        end

        function [x_model_backbone_417] = forward(this, x_model_backbone_411)
            if isdlarray(x_model_backbone_411)
                x_model_backbone_411 = stripdims(x_model_backbone_411);
            end
            x_model_backbone_411NumDims = 4;
            x_model_backbone_411 = netrai_fold0.ops.permuteInputVar(x_model_backbone_411, [4 3 1 2], 4);

            [x_model_backbone_417, x_model_backbone_417NumDims] = ReduceMeanGraph1084(this, x_model_backbone_411, x_model_backbone_411NumDims, true);
            x_model_backbone_417 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_417, [3 4 2 1], 4);

            x_model_backbone_417 = dlarray(single(x_model_backbone_417), 'SSCB');
        end

        function [x_model_backbone_417, x_model_backbone_417NumDims1086] = ReduceMeanGraph1084(this, x_model_backbone_411, x_model_backbone_411NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1085, x_model_backbone_411NumDims);
            xMean = mean(x_model_backbone_411, dims);
            x_model_backbone_417 = xMean;
            x_model_backbone_417NumDims = x_model_backbone_411NumDims;

            % Set graph output arguments
            x_model_backbone_417NumDims1086 = x_model_backbone_417NumDims;

        end

    end

end