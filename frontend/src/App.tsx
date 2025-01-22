import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Navigation } from './components/Navigation';
import { useComment } from './hooks/useComment';
import { Disc, FolderOpen, Folder, ChevronDown, ChevronRight, Users } from 'lucide-react';
import { CommentNode } from './types';

const queryClient = new QueryClient();

function AppContent() {
  const [path, setPath] = useState('');
  const [showClusters, setShowClusters] = useState(true);
  const [clusterMethod, setClusterMethod] = useState<'kmeans' | 'louvain' | 'raw'>('raw');
  const { data: comment, isLoading } = useComment(path);





  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-start">
      <header className="bg-white shadow w-full">
        <div className="max-w-7xl mx-auto py-4 px-4">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-gray-900">
              Comment Clusters
            </h1>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-6 px-4 w-full">
        <div className="flex">
          <div className="w-full max-w-2xl">
            <Navigation path={path} onNavigate={setPath} />
            {comment ? (() => {
              const commentRaw = comment.louvain_clusters.length === 0 && comment.kmeans_clusters.length === 0
              const commentKmeans = comment.kmeans_clusters.length > 0
              const commentLouvain = comment.louvain_clusters.length > 0

              const hasEnoughChildren = comment.config.min_cluster_size <= comment.raw_comments.length;

              const memberComments = clusterMethod === 'kmeans' ? comment.kmeans_clusters 
              : clusterMethod === 'louvain' ? comment.louvain_clusters
              : comment.raw_comments
              
              console.log('comment', comment, showClusters, hasEnoughChildren, memberComments)

              return (
                <div className="mt-4 p-4 bg-white rounded shadow">
                <div className="grid grid-cols-2 gap-2"> 
                  <div className='text-xs'> {comment.author}</div>
                  <div className='text-xs'> {comment.author}</div>
                </div> 
                <div 
                  className="text-sm text-gray-600" 
                  dangerouslySetInnerHTML={{__html: comment.text}} 
                />
                <div className="flex flex-col gap-2 mt-4">
                  {showClusters && hasEnoughChildren ? (<>
                    <select
                      value={clusterMethod}
                      onChange={(e) => setClusterMethod(e.target.value as 'kmeans' | 'louvain')}
                      className="px-3 py-2 text-sm font-medium text-gray-700 bg-white border rounded hover:bg-gray-50"
                      >
                      <option value="raw">Raw Comments</option>
                      <option value="kmeans">K-Means Clustering</option>
                      <option value="louvain">Louvain Clustering</option>
                    </select>
                    {clusterMethod === 'kmeans' ? (
                      comment.kmeans_clusters?.map((cluster, i) => (
                        <CommentCluster 
                          key={i}
                          index={i}
                          comment={cluster}
                          setPath={setPath}
                          path={path}
                          method={clusterMethod}
                        />
                      ))
                    ) : clusterMethod === 'louvain' ? (
                      comment.louvain_clusters?.map((cluster, i) => (
                        <CommentCluster 
                          key={i}
                          index={i}
                          comment={cluster}
                          setPath={setPath}
                          path={path}
                          method={clusterMethod}
                        />
                      ))
                    ) : (
                      comment.raw_comments?.map((child, i) => (
                        <ChildCommentComponent 
                          key={i}
                          comment={child}
                          setPath={setPath}
                          path={path}
                          index={i}
                        />
                      ))
                    )}
                  </>
                  ) : (
                    comment.raw_comments?.map((child, i) => (
                      <ChildCommentComponent 
                        key={i}
                        comment={child}
                        setPath={setPath}
                        path={path}
                        index={i}
                      />
                    ))
                  )}
                </div>
              </div>
              )
            })() : isLoading ? (
              <div>Loading...</div>
            ) : (
              <div>No comments found</div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

function CommentCluster({ 
  comment, 
  setPath, 
  path,
  method,
  index,
}: { 
  comment: CommentNode;
  setPath: (path: string) => void;
  path: string;
  method: 'kmeans' | 'louvain' | 'raw';
  index: number;
}) {

  const [isExpanded, setIsExpanded] = useState(false);
  const [showMembers, setShowMembers] = useState(false);

  const commentRaw = comment.louvain_clusters.length === 0 && comment.kmeans_clusters.length === 0
  const commentKmeans = comment.kmeans_clusters.length > 0
  const commentLouvain = comment.louvain_clusters.length > 0

  const hasEnoughChildren = comment.config.min_cluster_size <= comment.raw_comments.length;

  
  const clusters = method === 'kmeans' 
    ? comment.kmeans_clusters 
    : comment.louvain_clusters;

  const hasMembers = method === 'kmeans' ? commentKmeans : method === 'louvain' ? commentLouvain : false

  const memberComments = method === 'kmeans' ? comment.kmeans_clusters 
    : method === 'louvain' ? comment.louvain_clusters
    : comment.raw_comments


  const isRawNode = isExpanded || !hasMembers || !hasEnoughChildren;


  const pathElem = method === 'kmeans' ? `k${index}` : method === 'louvain' ? `l${index}` : `${index}`;

  console.log("comment", comment)

  return (
    <div className="p-3 bg-blue-50 rounded border border-blue-200">
      <div className="flex items-center gap-2">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="hover:bg-blue-100 rounded p-1"
        >
          {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </button>
        <div className="flex items-center gap-2 flex-1">
          <button
            onClick={() => {
              const newPath = path.length > 0 
                ? `${path}-${pathElem}` 
                : pathElem;
              setPath(newPath);
            }}
            className="hover:text-blue-600 p-1 rounded"
          >
            <Disc size={14} />
          </button>
          <div 
            className="text-sm text-gray-800 flex-1" 
            dangerouslySetInnerHTML={{__html: comment.text}} 
          />
        </div>
        {hasMembers && (
          <button
            onClick={() => setShowMembers(!showMembers)}
            className="hover:bg-blue-100 rounded p-1 flex items-center gap-1"
          >
            <Users size={16} />
            <span className="text-xs">
              {!isRawNode ? memberComments.length : 0}
            </span>
          </button>
        )}
      </div>
      {isExpanded && (
        comment.raw_comments.map((child, i) => (
          <ChildCommentComponent 
            key={i}
            comment={child}
            setPath={setPath}
            path={path}
            index={i}
          />
        ))
      )}
    </div>
  );
}




const CommentClusterComponent = ({
  comment,
  setPath,
  path,
  method
}: {
  comment: CommentNode;
  setPath: (path: string) => void;
  path: string;
  method: 'kmeans' | 'louvain';
}) => {



  return (<>
  </>)

}



// (
//   <div className="ml-6 mt-2 space-y-2">
//     {showMembers && (
//       <div className="space-y-2">
//         {memberComments.map((member, i) => (
//           <div 
//             key={i}
//             className="p-2 bg-white rounded border border-blue-100 flex items-start gap-2"
//           >
//             <button
//               onClick={() => setPath(member.route)}
//               className="hover:text-blue-600 p-1 rounded"
//             >
//               <Disc size={14} />
//             </button>
//             <div className="flex-1">
//               <div 
//                 className="text-sm text-gray-800" 
//                 dangerouslySetInnerHTML={{__html: member.text}} 
//               />
//               <div className="text-xs text-gray-500 mt-1">
//                 Confidence: {(member.confidence * 100).toFixed(1)}%
//               </div>
//             </div>
//           </div>
//         ))}
//       </div>
//     )}
//     {clusters && clusters.length > 0 && (
//       <div className="space-y-2">
//         {clusters.map((cluster, i) => (
//           <CommentCluster
//             key={i}
//             comment={cluster}
//             setPath={setPath}
//             path={path}
//             method={method}
//           />
//         ))}
//       </div>
//     )}
//   </div>
// )



const ChildCommentComponent = ({ 
  comment, 
  setPath, 
  path,
  index
}: { 
  comment: CommentNode;
  setPath: (path: string) => void;
  path: string;
  index: number;
}) =>  {
  return (
    <div className="p-2 bg-white rounded border border-gray-200 flex items-start gap-2">
      <button
        onClick={() => {
          const newPath = path.length > 0 
            ? `${path}-${index}` 
            : index.toString();
          setPath(newPath);
        }}
        className="hover:text-blue-600 p-1 rounded"
      >
        <Disc size={14} />
      </button>
      <div 
        className="text-sm text-gray-800 flex-1" 
        dangerouslySetInnerHTML={{__html: comment.text}} 
      />
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}