package catanai.stac;

import soc.game.SOCGame;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.robot.SOCRobotFactory;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * StacSettlers robot factory for the Catan AI bridge. Named in a STAC simulation config as
 * {@code Agent=1,<name>,class:catanai.stac.CatanAiStacFactory} (the class-name branch is part of the
 * Catan AI STAC match patch). Type flags are not supported: every brain is the same bridge brain.
 */
public class CatanAiStacFactory implements SOCRobotFactory
{
    public SOCRobotBrain getRobot(final SOCRobotClient cl, final SOCRobotParameters params, final SOCGame ga,
        final CappedQueue mq)
    {
        return new CatanAiStacBrain(cl, params, ga, mq);
    }

    public boolean isType(final String type)
    {
        return false;
    }

    public void setTypeFlag(final String type)
    {
        throw new UnsupportedOperationException("CatanAiStacFactory takes no type flags: " + type);
    }

    public void setTypeFlag(final String type, final String param)
    {
        throw new UnsupportedOperationException("CatanAiStacFactory takes no type flags: " + type);
    }
}
